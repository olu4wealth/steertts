package com.example.steertts.tts

import android.os.Bundle
import android.speech.tts.SynthesisCallback
import android.speech.tts.SynthesisRequest
import android.speech.tts.TextToSpeech
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.android.controller.ServiceController
import org.robolectric.annotation.Config
import java.lang.reflect.Method
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Runs the real TextToSpeechService lifecycle on the JVM.
 *
 * A device would be better (the framework actually binds the engine), but this container has no
 * usable /dev/kvm, so Robolectric drives onCreate plus the protected hooks instead. What it proves:
 * the language hooks answer with the framework's own constants, and a failed initialisation reaches
 * the caller as exactly one terminal callback - the failure mode that would otherwise leave every
 * app waiting forever on a synthesis request.
 *
 * The hooks are protected in TextToSpeechService and SteerTtsService is final, so they are invoked
 * reflectively rather than by widening the production API for tests.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class SteerTtsServiceTest {

    private lateinit var controller: ServiceController<SteerTtsService>
    private lateinit var service: SteerTtsService

    @Before
    fun setUp() {
        controller = Robolectric.buildService(SteerTtsService::class.java)
        service = controller.create().get()
    }

    @After
    fun tearDown() {
        controller.destroy()
    }

    private fun hook(name: String, vararg types: Class<*>): Method =
        SteerTtsService::class.java.getDeclaredMethod(name, *types).apply { isAccessible = true }

    private fun isLanguageAvailable(lang: String?, country: String?, variant: String?): Int =
        hook(
            "onIsLanguageAvailable",
            String::class.java,
            String::class.java,
            String::class.java,
        ).invoke(service, lang, country, variant) as Int

    private fun synthesize(text: String, callback: SynthesisCallback) {
        hook("onSynthesizeText", SynthesisRequest::class.java, SynthesisCallback::class.java)
            .invoke(service, SynthesisRequest(text, Bundle()), callback)
    }

    @Test
    fun `language hooks claim only generic English`() {
        assertEquals(TextToSpeech.LANG_AVAILABLE, isLanguageAvailable("eng", "", ""))
        // Case-insensitive, and a country request still resolves to the generic voice.
        assertEquals(TextToSpeech.LANG_AVAILABLE, isLanguageAvailable("ENG", "USA", "coach"))
        assertEquals(TextToSpeech.LANG_NOT_SUPPORTED, isLanguageAvailable("deu", "DEU", ""))
        assertEquals(TextToSpeech.LANG_NOT_SUPPORTED, isLanguageAvailable(null, null, null))

        val language = hook("onGetLanguage").invoke(service) as Array<*>
        assertArrayEquals(arrayOf("eng", "", ""), language)

        assertEquals(
            TextToSpeech.LANG_AVAILABLE,
            hook("onLoadLanguage", String::class.java, String::class.java, String::class.java)
                .invoke(service, "eng", "", "") as Int,
        )
    }

    @Test
    fun `synthesis without model assets reports one error and never starts`() {
        val callback = RecordingCallback()

        synthesize("hello, world", callback)

        // No audio package is installed, so initialisation fails; the framework must be told once,
        // with error() only (calling error() after start() is illegal) and never done().
        assertEquals(listOf("error(${TextToSpeech.ERROR})"), callback.events)
        assertFalse(callback.hasStarted())
        assertTrue(callback.hasFinished())
    }

    @Test
    fun `blank text is rejected as an invalid request`() {
        val callback = RecordingCallback()

        synthesize("   ", callback)

        assertEquals(listOf("error(${TextToSpeech.ERROR_INVALID_REQUEST})"), callback.events)
        assertFalse(callback.hasStarted())
    }

    @Test
    fun `no rebuild is attempted before the backoff elapses`() {
        val first = RecordingCallback()
        synthesize("AB", first)
        assertEquals(listOf("error(${TextToSpeech.ERROR})"), first.events)

        // The model became available, but the default backoff has not elapsed, so this request must
        // fail fast instead of paying for another initialisation attempt.
        service.engineFactory = { stubEngine() }

        val second = RecordingCallback()
        synthesize("AB", second)

        assertEquals(listOf("error(${TextToSpeech.ERROR})"), second.events)
        assertFalse(second.hasStarted())
    }

    @Test
    fun `a later request rebuilds the engine and speaks`() {
        val failing = RecordingCallback()
        synthesize("AB", failing)
        assertEquals(listOf("error(${TextToSpeech.ERROR})"), failing.events)

        // Model installed while the process is alive: past the backoff window, the engine recovers.
        val engine = stubEngine()
        service.engineFactory = { engine }
        service.retryBackoffMillis = 0

        val callback = RecordingCallback()
        synthesize("AB", callback)

        assertEquals("start(24000,2,1)", callback.events.first())
        assertEquals("audio", callback.events[1])
        assertEquals("done", callback.events.last())
        assertEquals("132 stub samples as 16-bit mono", 132 * 2, callback.audioBytes)
        assertTrue(callback.hasFinished())
    }

    /** A real session over the echo stub graph, so the service's audio path can actually run. */
    private fun stubEngine(): SteerTtsInference {
        val bytes = checkNotNull(javaClass.getResourceAsStream("/steer/echo.onnx")) {
            "Missing test fixture /steer/echo.onnx"
        }.use { it.readBytes() }
        val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        buffer.put(bytes)
        buffer.rewind()
        return SteerTtsInference.createFromParts(
            modelBuffer = buffer,
            tokenizer = SteerTokenizer.fromSymbolTable("A\nB\n"),
            speakerEmbedding = FloatArray(8),
        )
    }

    /** Records the callback sequence so the terminal-callback contract is assertable. */
    private class RecordingCallback : SynthesisCallback {
        val events = mutableListOf<String>()
        var audioBytes = 0
        private var started = false
        private var finished = false

        override fun getMaxBufferSize(): Int = 4_096

        override fun start(sampleRateInHz: Int, audioFormat: Int, channelCount: Int): Int {
            events += "start($sampleRateInHz,$audioFormat,$channelCount)"
            started = true
            return TextToSpeech.SUCCESS
        }

        override fun audioAvailable(buffer: ByteArray, offset: Int, length: Int): Int {
            audioBytes += length
            if (events.lastOrNull() != "audio") events += "audio"
            return TextToSpeech.SUCCESS
        }

        override fun done(): Int {
            events += "done"
            finished = true
            return TextToSpeech.SUCCESS
        }

        override fun error() {
            events += "error()"
            finished = true
        }

        override fun error(errorCode: Int) {
            events += "error($errorCode)"
            finished = true
        }

        override fun hasStarted(): Boolean = started

        override fun hasFinished(): Boolean = finished
    }
}
