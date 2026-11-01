package com.example.steertts.tts

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SteeringParamsTest {

    @Test
    fun `unspecified requests nothing`() {
        val params = SteeringParams.UNSPECIFIED
        assertTrue(params.isEmpty)
        assertTrue(params.maskArray().all { it == 0f })
        assertTrue(params.ctrlArray().all { it == 0f })
    }

    @Test
    fun `neutral marks every dimension as requested`() {
        val mask = SteeringParams.NEUTRAL.maskArray()
        assertEquals(Control.entries.size, mask.size)
        assertTrue(mask.all { it == 1f })
        assertTrue(SteeringParams.NEUTRAL.ctrlArray().all { it == 0f })
    }

    @Test
    fun `pitch is mapped onto f0_med only`() {
        val params = SteeringParams.fromSystemRequest(pitch = 2f, speechRate = 1f) // one octave up
        val ctrl = params.ctrlArray()
        val mask = params.maskArray()

        assertEquals(4f, ctrl[Control.F0_MED.ordinal], 1e-4f)
        assertEquals(1f, mask[Control.F0_MED.ordinal], 0f)
        // Everything else must stay unspecified, not "explicitly neutral".
        Control.entries.filter { it != Control.F0_MED }.forEach {
            assertEquals("${it.wireName} should be unmasked", 0f, mask[it.ordinal], 0f)
            assertEquals(0f, ctrl[it.ordinal], 1e-6f)
        }
    }

    @Test
    fun `speech rate is mapped onto rate only`() {
        val ctrl = SteeringParams.fromSystemRequest(pitch = 1f, speechRate = 2f).ctrlArray()
        assertEquals(kotlin.math.ln(2f) * 3f, ctrl[Control.RATE.ordinal], 1e-4f)
        assertEquals(0f, ctrl[Control.F0_MED.ordinal], 1e-6f)
    }

    @Test
    fun `neutral system values request nothing`() {
        assertTrue(SteeringParams.fromSystemRequest(1f, 1f).isEmpty)
    }

    @Test
    fun `values are clamped to the documented range`() {
        assertEquals(5f, SteeringParams.of(Control.LOUD to 99f).ctrlArray()[Control.LOUD.ordinal], 0f)
        assertEquals(-5f, SteeringParams.of(Control.LOUD to -99f).ctrlArray()[Control.LOUD.ordinal], 0f)
        // 8 octaves of pitch would leave the range, so it must clamp.
        assertEquals(-5f, SteeringParams.fromSystemRequest(0.01f, 1f).ctrlArray()[Control.F0_MED.ordinal], 0f)
    }

    @Test
    fun `with accumulates dimensions`() {
        val params = SteeringParams.of(Control.RATE to 1f).with(Control.LOUD, -2f)
        val ctrl = params.ctrlArray()
        assertEquals(1f, ctrl[Control.RATE.ordinal], 1e-6f)
        assertEquals(-2f, ctrl[Control.LOUD.ordinal], 1e-6f)
        assertEquals(2, params.maskArray().count { it == 1f })
    }

    @Test
    fun `array order matches the enum order`() {
        val expected = Control.entries.map { it.wireName }
        assertEquals(
            listOf(
                "rate", "loud", "f0_med", "f0_iqr", "f0_end", "pause",
                "tilt", "voiced", "flat", "centroid", "cpps", "h1h2",
            ),
            expected,
        )
        assertEquals(12, Control.entries.size)
    }
}
