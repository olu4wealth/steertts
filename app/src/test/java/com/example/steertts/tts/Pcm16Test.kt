package com.example.steertts.tts

import org.junit.Assert.assertEquals
import org.junit.Test

class Pcm16Test {

    private fun encode(samples: FloatArray) =
        Pcm16.encode(samples, offset = 0, count = samples.size, fadeIn = false, fadeOut = false)

    @Test
    fun `two bytes per sample little endian`() {
        val pcm = encode(floatArrayOf(0f, 1f, -1f))
        assertEquals(6, pcm.size)

        assertEquals(0, pcm[0].toInt())
        assertEquals(0, pcm[1].toInt())

        // 32767 == 0x7FFF
        assertEquals(0xFF.toByte(), pcm[2])
        assertEquals(0x7F.toByte(), pcm[3])

        // -32767 == 0x8001 -> little endian 0x01, 0x80
        assertEquals(0x01.toByte(), pcm[4])
        assertEquals(0x80.toByte(), pcm[5])
    }

    @Test
    fun `out of range samples clip instead of wrapping`() {
        val pcm = encode(floatArrayOf(1.5f, -1.5f))
        assertEquals(0xFF.toByte(), pcm[0])
        assertEquals(0x7F.toByte(), pcm[1])
        assertEquals(0x01.toByte(), pcm[2])
        assertEquals(0x80.toByte(), pcm[3])
    }

    @Test
    fun `fade in starts silent and leaves the tail untouched`() {
        val samples = FloatArray(1_000) { 1f }
        val pcm = Pcm16.encode(samples, offset = 0, count = samples.size, fadeIn = true, fadeOut = false)

        assertEquals(0, pcm[0].toInt())
        assertEquals(0, pcm[1].toInt())

        val midIndex = 500 * 2
        assertEquals(0xFF.toByte(), pcm[midIndex])
        assertEquals(0x7F.toByte(), pcm[midIndex + 1])
    }

    @Test
    fun `fade out ramps the tail down and keeps the length exact`() {
        val samples = FloatArray(1_000) { 1f }
        val pcm = Pcm16.encode(samples, offset = 0, count = samples.size, fadeIn = false, fadeOut = true)
        assertEquals(2_000, pcm.size)

        fun sampleAt(index: Int): Int =
            ((pcm[index * 2].toInt() and 0xFF) or (pcm[index * 2 + 1].toInt() shl 8)).toShort().toInt()

        assertEquals(32767, sampleAt(500))                 // untouched before the ramp
        assert(sampleAt(940) < sampleAt(900))              // decreasing inside the ramp
        // A linear ramp of FADE_SAMPLES ends at 1/FADE_SAMPLES of full scale, not exactly zero.
        assert(kotlin.math.abs(sampleAt(999)) < 400)
    }

    @Test
    fun `offset window is honoured`() {
        val samples = floatArrayOf(0f, 0f, 1f, 1f)
        val pcm = Pcm16.encode(samples, offset = 2, count = 2, fadeIn = false, fadeOut = false)
        assertEquals(4, pcm.size)
        assertEquals(0xFF.toByte(), pcm[0])
        assertEquals(0x7F.toByte(), pcm[1])
    }
}
