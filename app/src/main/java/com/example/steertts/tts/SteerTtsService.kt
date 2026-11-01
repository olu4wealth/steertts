package com.example.steertts.tts

import android.media.AudioFormat
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import android.speech.tts.TextToSpeechService
import android.util.Log
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.TimeoutException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.min

/**
 * System-wide TTS engine. The framework binds this, calls onSynthesizeText() off the main thread,
 * and consumes the PCM we push into the SynthesisCallback.
 */
class SteerTtsService : TextToSpeechService() {

    private val engine = AtomicReference<SteerTtsInference?>(null)
    private val initFailure = AtomicReference<Throwable?>(null)
    private val initInFlight = AtomicBoolean(false)
    private val firstAttemptDone = CountDownLatch(1)
    private val lastFailureAtNanos = AtomicLong(0)

    /**
     * Minimum wait before a failed initialisation is retried, so a missing model is not re-read on
     * every utterance. Internal so tests can drive recovery without waiting.
     */
    internal var retryBackoffMillis: Long = INIT_RETRY_BACKOFF_MILLIS

    /** Test seam: production loads the packaged assets. */
    internal var engineFactory: () -> SteerTtsInference = { SteerTtsInference.create(assets) }

    /**
     * onSynthesizeText already runs on a binder thread, not the UI thread. A single-thread executor
     * additionally serializes utterances: one ORT run at a time, bounded memory, ordered callbacks.
     */
    private val inferenceExecutor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "steertts-inference").apply {
            isDaemon = true
            priority = Thread.NORM_PRIORITY - 1
        }
    }

    private val stopRequested = AtomicBoolean(false)
    private val currentSink = AtomicReference<SynthesisSink?>(null)

    // ------------------------------------------------------------- lifecycle

    override fun onCreate() {
        super.onCreate()
        // onCreate runs on the main thread: never load the model here.
        inferenceExecutor.execute { startInitialization() }
    }

    /**
     * Runs one initialisation attempt and never throws. Returns false when an attempt is already in
     * flight, so concurrent requests cannot create two sessions.
     */
    private fun startInitialization(): Boolean {
        if (!initInFlight.compareAndSet(false, true)) return false
        try {
            val inference = engineFactory()
            engine.set(inference)
            initFailure.set(null)
            lastFailureAtNanos.set(0)
            Log.i(TAG, "SteerTTS ready: ${inference.describe()}")
        } catch (t: Throwable) {
            // Missing assets, corrupt graph, ORT init failure: remember it, report per utterance,
            // and allow a later request to try again once the backoff has elapsed.
            initFailure.set(t)
            lastFailureAtNanos.set(System.nanoTime())
            Log.e(TAG, "Initialisation failed; a later request will retry", t)
        } finally {
            initInFlight.set(false)
            firstAttemptDone.countDown()
        }
        return true
    }

    override fun onDestroy() {
        engine.getAndSet(null)?.close()
        inferenceExecutor.shutdownNow()
        super.onDestroy()
    }

    // ------------------------------------------- required TextToSpeechService hooks
    // NOTE: the language hooks take (lang, country, variant) strings, not a Locale.

    override fun onIsLanguageAvailable(lang: String?, country: String?, variant: String?): Int {
        if (lang.isNullOrEmpty() || !lang.equals(LANGUAGE, ignoreCase = true)) {
            return TextToSpeech.LANG_NOT_SUPPORTED
        }
        // Claim only the generic voice: LANG_COUNTRY_AVAILABLE would promise an accent-specific
        // voice that this engine does not actually have.
        return TextToSpeech.LANG_AVAILABLE
    }

    override fun onGetLanguage(): Array<String> = arrayOf(LANGUAGE, "", "")

    override fun onLoadLanguage(lang: String?, country: String?, variant: String?): Int =
        onIsLanguageAvailable(lang, country, variant)

    override fun onStop() {
        stopRequested.set(true)
        currentSink.get()?.abandon() // unblocks the streaming loop at the next chunk
    }

    // ------------------------------------------------------------- synthesis

    override fun onSynthesizeText(request: SynthesisRequest, callback: SynthesisCallback) {
        stopRequested.set(false)
        val sink = SynthesisSink(callback)
        currentSink.set(sink)

        // getCharSequenceText() is public since API 21; getText() is deprecated and drops spans.
        val text = request.charSequenceText?.toString()?.trim().orEmpty()
        if (text.isEmpty()) {
            sink.fail(TextToSpeech.ERROR_INVALID_REQUEST)
            currentSink.compareAndSet(sink, null)
            return
        }

        try {
            val inference = engineOrRetry()
            if (inference == null) {
                Log.e(TAG, "Engine unavailable: ${initFailure.get()?.message}")
                sink.fail()
                return
            }

            // Blocks this binder thread until the queued synthesis finishes, so the framework sees
            // one utterance at a time and callback ordering is preserved.
            inferenceExecutor.submit(
                Runnable { synthesizeBlocking(inference, request, text, sink) }
            ).get(SYNTHESIS_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        } catch (e: TimeoutException) {
            Log.e(TAG, "Synthesis timed out after ${SYNTHESIS_TIMEOUT_SECONDS}s", e)
            sink.fail()
        } catch (e: ExecutionException) {
            Log.e(TAG, "Synthesis failed", e.cause ?: e)
            sink.fail()
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
            sink.fail()
        } finally {
            currentSink.compareAndSet(sink, null)
        }
    }

    private fun synthesizeBlocking(
        inference: SteerTtsInference,
        request: SynthesisRequest,
        text: String,
        sink: SynthesisSink,
    ) {
        // getPitch()/getSpeechRate() are ints in hundredths: 100 == 1.0x.
        val steering = SteeringParams.fromSystemRequest(
            pitch = request.pitch / 100f,
            speechRate = request.speechRate / 100f,
        )

        val samples = inference.synthesize(text = text, steering = steering)

        if (stopRequested.get()) {
            sink.abandon()
            return
        }
        if (samples.isEmpty()) {
            Log.w(TAG, "Model produced no audio for ${text.length} characters")
            sink.fail()
            return
        }
        if (!sink.start(inference.sampleRate)) {
            Log.w(TAG, "AudioTrack refused start()")
            return
        }

        // Stream in ~200 ms chunks instead of one huge audioAvailable() call.
        var offset = 0
        while (offset < samples.size) {
            if (stopRequested.get()) {
                sink.abandon()
                return
            }
            val count = min(CHUNK_SAMPLES, samples.size - offset)
            val pcm = Pcm16.encode(
                samples = samples,
                offset = offset,
                count = count,
                fadeIn = offset == 0,
                fadeOut = offset + count >= samples.size,
            )
            if (!sink.write(pcm)) {
                sink.abandon() // consumer went away (AudioTrack error or stop)
                return
            }
            offset += count
        }
        sink.done()
    }

    /**
     * Returns a live session, or null when none is available for this request.
     *
     * The first attempt is always awaited, because onCreate already queued it. After a failure the
     * session is rebuilt at most once per backoff window: that lets the engine recover when the
     * model is installed while the process is alive, without re-reading a missing asset on every
     * single utterance.
     */
    private fun engineOrRetry(): SteerTtsInference? {
        engine.get()?.let { return it }

        if (!awaitFirstAttempt()) {
            Log.e(TAG, "Initial initialisation did not finish within ${INIT_TIMEOUT_SECONDS}s")
            return engine.get()
        }
        engine.get()?.let { return it }

        val failedAt = lastFailureAtNanos.get()
        val sinceFailureMs = (System.nanoTime() - failedAt) / 1_000_000
        if (failedAt != 0L && sinceFailureMs < retryBackoffMillis) {
            Log.w(
                TAG,
                "Initialisation failed ${sinceFailureMs}ms ago; " +
                    "next attempt in ${retryBackoffMillis - sinceFailureMs}ms",
            )
            return null
        }

        // The request that triggers the rebuild pays for it, on the inference thread and with the
        // same bound as the first attempt, so a hanging load cannot pin the binder thread.
        return try {
            inferenceExecutor
                .submit(Runnable { startInitialization() })
                .get(INIT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
            engine.get()
        } catch (e: TimeoutException) {
            Log.e(TAG, "Re-initialisation timed out after ${INIT_TIMEOUT_SECONDS}s", e)
            null
        } catch (e: ExecutionException) {
            Log.e(TAG, "Re-initialisation failed", e.cause ?: e)
            null
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
            null
        }
    }

    private fun awaitFirstAttempt(): Boolean = try {
        firstAttemptDone.await(INIT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
    } catch (e: InterruptedException) {
        Thread.currentThread().interrupt()
        false
    }

    /**
     * Guarantees exactly one terminal callback and never calls done() without a successful start();
     * both are required by the SynthesisCallback contract.
     */
    private class SynthesisSink(private val callback: SynthesisCallback) {
        private var started = false
        private var terminal = false

        fun start(sampleRate: Int): Boolean {
            if (terminal) return false
            started = callback.start(
                sampleRate,
                AudioFormat.ENCODING_PCM_16BIT,
                1,
            ) == TextToSpeech.SUCCESS
            if (!started) terminal = true
            return started
        }

        fun write(pcm: ByteArray): Boolean {
            if (!started || terminal) return false
            val max = callback.maxBufferSize.coerceAtLeast(1)
            var offset = 0
            while (offset < pcm.size) {
                val count = min(max, pcm.size - offset)
                if (callback.audioAvailable(pcm, offset, count) != TextToSpeech.SUCCESS) {
                    terminal = true
                    return false
                }
                offset += count
            }
            return true
        }

        fun done() {
            if (started && !terminal) {
                terminal = true
                callback.done()
            }
        }

        fun fail(errorCode: Int = TextToSpeech.ERROR) {
            if (!started && !terminal) { // error() is only legal before start()
                terminal = true
                runCatching { callback.error(errorCode) }
            }
        }

        fun abandon() {
            // Stopped by the framework: the utterance is already cancelled, so do not call done().
            terminal = true
        }
    }

    companion object {
        private const val TAG = "SteerTtsService"
        private const val LANGUAGE = "eng"
        private const val INIT_TIMEOUT_SECONDS = 20L
        private const val SYNTHESIS_TIMEOUT_SECONDS = 120L
        private const val INIT_RETRY_BACKOFF_MILLIS = 15_000L
        private const val CHUNK_SAMPLES = 4_800 // 200 ms at 24 kHz
    }
}
