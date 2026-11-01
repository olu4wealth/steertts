package com.example.steertts.tts

import kotlin.math.ln
import kotlin.math.log2

/**
 * The 12 utterance-level acoustic controls of SteerTTS. Values are z-scores against the selected
 * voice's own neutral speech, in pooled within-speaker SD units, valid in [-5, 5].
 *
 * Declaration order is the wire order of the graph's `ctrl` and `cmask` tensors.
 */
enum class Control(val wireName: String, val label: String) {
    RATE("rate", "articulation rate"),
    LOUD("loud", "speech level"),
    F0_MED("f0_med", "median pitch"),
    F0_IQR("f0_iqr", "pitch range"),
    F0_END("f0_end", "final pitch slope"),
    PAUSE("pause", "pause share"),
    TILT("tilt", "spectral tilt"),
    VOICED("voiced", "voiced share"),
    FLAT("flat", "spectral flatness"),
    CENTROID("centroid", "spectral centroid"),
    CPPS("cpps", "cepstral peak prominence"),
    H1H2("h1h2", "H1-H2"),
}

/**
 * An immutable steering request.
 *
 * `ctrl` carries the requested z-scores; `mask` marks which dims were actually asked for.
 * A dim that is left out is *unspecified* (the model infers it from voice, tag and text), which
 * is not the same as sending 0.0 — 0.0 means "this voice's neutral value".
 */
class SteeringParams private constructor(private val requested: Map<Control, Float>) {

    val isEmpty: Boolean get() = requested.isEmpty()

    fun ctrlArray(dim: Int = Control.entries.size): FloatArray =
        FloatArray(dim) { index ->
            Control.entries.getOrNull(index)?.let { requested[it] } ?: 0f
        }

    fun maskArray(dim: Int = Control.entries.size): FloatArray =
        FloatArray(dim) { index ->
            val control = Control.entries.getOrNull(index)
            if (control != null && requested.containsKey(control)) 1f else 0f
        }

    fun with(control: Control, z: Float): SteeringParams =
        SteeringParams(requested + (control to z.coerceIn(Z_MIN, Z_MAX)))

    override fun toString(): String =
        requested.entries.joinToString(", ", "{", "}") { "${it.key.wireName}=${it.value}" }

    companion object {
        const val Z_MIN = -5f
        const val Z_MAX = 5f

        /** Nothing requested: the model decides every control. */
        val UNSPECIFIED = SteeringParams(emptyMap())

        /** Explicitly neutral delivery: every dim requested at the voice's mean. */
        val NEUTRAL = SteeringParams(Control.entries.associateWith { 0f })

        fun of(vararg zs: Pair<Control, Float>): SteeringParams =
            SteeringParams(zs.associate { (control, z) -> control to z.coerceIn(Z_MIN, Z_MAX) })

        // Tuning knobs, not physics: SteerTTS delivers roughly half of a requested z, so these
        // are sized to land near the requested perceptual change while staying inside [-5, 5].
        private const val Z_PER_OCTAVE = 4f
        private const val Z_PER_LOG_RATE = 3f

        /**
         * Bridges android.speech.tts.SynthesisRequest onto SteerTTS controls.
         * [pitch] and [speechRate] are multipliers where 1.0 is neutral.
         */
        fun fromSystemRequest(pitch: Float, speechRate: Float): SteeringParams {
            var params = UNSPECIFIED
            if (pitch > 0f && pitch != 1f) {
                params = params.with(Control.F0_MED, log2(pitch) * Z_PER_OCTAVE)
            }
            if (speechRate > 0f && speechRate != 1f) {
                params = params.with(Control.RATE, ln(speechRate) * Z_PER_LOG_RATE)
            }
            return params
        }
    }
}
