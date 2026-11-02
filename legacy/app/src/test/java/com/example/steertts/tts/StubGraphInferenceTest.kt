package com.example.steertts.tts

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The only test that executes the real ONNX path: a stub graph echoes every steering input back
 * out as its "audio", so we can assert the exact numbers SteerTTS would receive.
 *
 * The genuine weights are licence-gated (CC BY-NC 4.0) and are not in this repository, so this is
 * as close to end-to-end as the tensor wiring can be checked without them.
 *
 * Fixtures: `src/test/resources/steer/echo.onnx` and `narrow.onnx` (built by tools/gen_stub_models.py)
 *   audio[0, 0..11] = ctrl        the 12 z-scores as sent
 *   audio[0, 12]    = sum(cmask)  how many dims were actually requested
 *   audio[0, 13]    = n           phone count
 *   audio[0, 14]    = tag
 *   audio[0, 15]    = sum(spk)
 */
class StubGraphInferenceTest {

    private val tokenizer = SteerTokenizer.fromSymbolTable("A\nB\n")

    private fun open(model: String): SteerTtsInference {
        val bytes = checkNotNull(javaClass.getResourceAsStream("/steer/$model.onnx")) {
            "Missing test fixture /steer/$model.onnx"
        }.use { it.readBytes() }
        // Same contract as production: a direct buffer that stays referenced for the session's life.
        val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        buffer.put(bytes)
        buffer.rewind()
        return SteerTtsInference.createFromParts(
            modelBuffer = buffer,
            tokenizer = tokenizer,
            speakerEmbedding = floatArrayOf(0.5f, 0.25f, 0f, 0f, 0f, 0f, 0f, 0f),
        )
    }

    @Test
    fun `requested controls reach the graph and the mask counts them`() {
        open("echo").use { tts ->
            val out = tts.synthesize(
                "AB",
                SteeringParams.of(Control.RATE to 2.5f, Control.F0_MED to -1.5f),
            )

            assertEquals(132, out.size)
            assertEquals(2.5f, out[Control.RATE.ordinal], 1e-6f)
            assertEquals(-1.5f, out[Control.F0_MED.ordinal], 1e-6f)
            assertEquals(0f, out[Control.PAUSE.ordinal], 1e-6f) // unrequested dims stay 0.0
            assertEquals("cmask must flag exactly the 2 requested dims", 2f, out[12], 1e-6f)
            assertEquals(
                "n must carry the token count",
                tokenizer.encode("AB").size.toFloat(),
                out[13],
                1e-6f,
            )
            assertEquals("default tag", 0f, out[14], 1e-6f)
            assertEquals("speaker row must reach the graph", 0.75f, out[15], 1e-6f)
        }
    }

    @Test
    fun `unspecified steering sends zeros with an empty mask`() {
        open("echo").use { tts ->
            val out = tts.synthesize("AB")

            for (control in Control.entries) {
                assertEquals(0f, out[control.ordinal], 1e-6f)
            }
            // 0.0 with mask 0 means "model decides"; 0.0 with mask 1 would mean "neutral", so an
            // accidentally all-ones mask would silently change the delivery of every utterance.
            assertEquals(0f, out[12], 1e-6f)
        }
    }

    @Test
    fun `neutral steering requests every dim explicitly`() {
        open("echo").use { tts ->
            val out = tts.synthesize("AB", SteeringParams.NEUTRAL)

            assertEquals(0f, out[Control.RATE.ordinal], 1e-6f)
            assertEquals(Control.entries.size.toFloat(), out[12], 1e-6f)
        }
    }

    @Test
    fun `system pitch maps onto median pitch as a log ratio`() {
        open("echo").use { tts ->
            // One octave up: log2(2) * Z_PER_OCTAVE = 4, and rate 1.0 stays unspecified.
            val out = tts.synthesize(
                "AB",
                SteeringParams.fromSystemRequest(pitch = 2f, speechRate = 1f),
            )

            assertEquals(4f, out[Control.F0_MED.ordinal], 1e-6f)
            assertEquals(0f, out[Control.RATE.ordinal], 1e-6f)
            assertEquals(1f, out[12], 1e-6f)
        }
    }

    @Test
    fun `a graph returning pooled features instead of a waveform is rejected`() {
        open("narrow").use { tts ->
            val error = assertThrows(SteerTtsModelContractException::class.java) {
                tts.synthesize("AB")
            }
            // This is the guard that keeps an unpacked enc/dec/voc package from being played back
            // as noise: its first output is not a waveform.
            assertTrue(error.message!!.contains("not a waveform"))
        }
    }
}
