package com.example.steertts.tts

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.res.AssetManager
import android.util.Log
import org.json.JSONObject
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class SteerTtsMissingAssetException(path: String, cause: Throwable? = null) :
    IOException("Missing asset '$path' under src/main/assets", cause)

class SteerTtsModelContractException(message: String) : IllegalStateException(message)

/**
 * Owns the ONNX Runtime session and turns text into float PCM.
 *
 * Threading: one serialized caller keeps peak memory bounded (activations for a long sentence are
 * large) and makes onStop() semantics predictable. The lock also guards close().
 *
 * Lifecycle: create once, keep for the process lifetime, close() in onDestroy().
 */
class SteerTtsInference private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    @Suppress("unused") private val modelBuffer: ByteBuffer,
    private val tokenizer: SteerTokenizer,
    private val speakerEmbedding: FloatArray,
    private val controlDim: Int,
    private val guidance: Float,
    val sampleRate: Int,
    private val modelName: String,
) : AutoCloseable {

    private val runLock = ReentrantLock()
    private var closed = false

    /** @param tagId style tag id; 0 is "none" in the SteerTTS tag table. */
    fun synthesize(
        text: String,
        steering: SteeringParams = SteeringParams.UNSPECIFIED,
        pace: Float = 1f,
        tagId: Long = 0L,
    ): FloatArray {
        val ids = tokenizer.encode(text)
        if (ids.isEmpty()) return FloatArray(0)

        runLock.withLock {
            check(!closed) { "SteerTtsInference is closed" }
            val bag = TensorBag(env)
            try {
                val declared = session.inputNames

                // 1. phoneme ids [1, N]
                val idName = findInput(declared, "ids", "input_ids", "phonemes", "tokens")
                    ?: throw SteerTtsModelContractException(
                        "Graph '$modelName' declares inputs $declared; " +
                            "none of ids/input_ids/phonemes/tokens is present"
                    )
                bag.putLong(idName, ids, longArrayOf(1, ids.size.toLong()))

                // 2. true phone count [1] (the edge graphs pad sequences to a multiple of 64)
                findInput(declared, "n", "lengths", "input_lengths", "n_phones")
                    ?.let { bag.putLong(it, longArrayOf(ids.size.toLong()), longArrayOf(1)) }

                // 3. speaker vector [1, D]
                findInput(declared, "spk", "speaker", "speaker_embedding")?.let {
                    bag.putFloat(it, speakerEmbedding, longArrayOf(1, speakerEmbedding.size.toLong()))
                }

                // 4. style tag [1]
                findInput(declared, "tag", "style")?.let {
                    bag.putLong(it, longArrayOf(tagId), longArrayOf(1))
                }

                // 5. the 12 acoustic steering controls [1, 12]
                val ctrlName = findInput(declared, "ctrl", "controls", "steer")
                if (ctrlName != null) {
                    bag.putFloat(
                        ctrlName,
                        steering.ctrlArray(controlDim),
                        longArrayOf(1, controlDim.toLong()),
                    )
                } else {
                    Log.w(TAG, "Graph has no ctrl input; acoustic steering is ignored")
                }

                // 6. the availability mask [1, 12]. Without it an all-zero ctrl would mean
                // "neutral", not "unspecified", so send it whenever the graph declares it.
                findInput(declared, "cmask", "ctrl_mask", "mask", "steer_mask")?.let {
                    bag.putFloat(
                        it,
                        steering.maskArray(controlDim),
                        longArrayOf(1, controlDim.toLong()),
                    )
                }

                // Optional scalars.
                findInput(declared, "guidance", "s")?.let {
                    bag.putFloat(it, floatArrayOf(guidance), longArrayOf(1))
                }
                findInput(declared, "pace")?.let {
                    bag.putFloat(it, floatArrayOf(pace), longArrayOf(1))
                }

                val result = try {
                    session.run(bag.map)
                } catch (e: OrtException) {
                    throw SteerTtsModelContractException(
                        "ONNX Runtime rejected the input map: ${e.message}"
                    )
                }

                try {
                    val tensor = result.get(0) as? OnnxTensor
                        ?: throw SteerTtsModelContractException(
                            "Output 0 of '$modelName' is not a tensor"
                        )
                    val shape = (tensor.info as TensorInfo).shape
                    val last = shape.lastOrNull() ?: 0L
                    if (last < MIN_AUDIO_SAMPLES) {
                        throw SteerTtsModelContractException(
                            "Output '${session.outputNames.first()}' has shape " +
                                "${shape.contentToString()}, which is not a waveform. The published " +
                                "edge package ships enc/dec/voc graphs; fuse them, or implement the " +
                                "duration rounding / length regulation / iSTFT glue."
                        )
                    }
                    return readFloats(tensor)
                } finally {
                    result.close()
                }
            } finally {
                bag.close() // always release input tensors, including on failure
            }
        }
    }

    fun describe(): String =
        "model=$modelName sampleRate=$sampleRate controlDim=$controlDim " +
            "in=${session.inputNames.joinToString(",")} out=${session.outputNames.joinToString(",")}"

    override fun close() {
        runLock.withLock {
            if (closed) return
            closed = true
            runCatching { session.close() }
            // OrtEnvironment is a process-wide singleton from getEnvironment(): never close it
            // here, or every later inference in the process fails.
        }
    }

    companion object {
        private const val TAG = "SteerTtsInference"
        private const val MIN_AUDIO_SAMPLES = 64L
        private const val DEFAULT_CONTROL_DIM = 12
        private const val DEFAULT_SAMPLE_RATE = 24_000

        fun create(assets: AssetManager, assetRoot: String = "steer"): SteerTtsInference {
            val config = readJson(assets, "$assetRoot/config.json")
            val modelFile = "$assetRoot/" + config.optString("model", "model.onnx")
            val tokenFile = "$assetRoot/" + config.optString("tokens", "tokens.txt")
            val speakerFile = "$assetRoot/" + config.optString("speakers", "voices.safetensors")
            val speakerTensor = config.optString("speaker_tensor", "s")
            val speakerIndex = config.optInt("speaker_index", 0)
            val controlDim = config.optInt("control_dim", DEFAULT_CONTROL_DIM)
            val sampleRate = config.optInt("sample_rate", DEFAULT_SAMPLE_RATE)
            val guidance = config.optDouble("guidance", 1.0).toFloat()

            val tokenizer = SteerTokenizer.load(assets, tokenFile)
            val speaker = Safetensors.readFloatRow(assets, speakerFile, speakerTensor, speakerIndex)
            val modelBuffer = readModelBuffer(assets, modelFile)

            return createFromParts(
                modelBuffer = modelBuffer,
                tokenizer = tokenizer,
                speakerEmbedding = speaker,
                controlDim = controlDim,
                sampleRate = sampleRate,
                guidance = guidance,
                modelName = modelFile,
            )
        }

        /**
         * Builds the session from already-loaded pieces. Reading assets is the only Android part
         * of setup, so keeping it out of this path lets a JVM test execute a real graph.
         */
        internal fun createFromParts(
            modelBuffer: ByteBuffer,
            tokenizer: SteerTokenizer,
            speakerEmbedding: FloatArray,
            controlDim: Int = DEFAULT_CONTROL_DIM,
            sampleRate: Int = DEFAULT_SAMPLE_RATE,
            guidance: Float = 1f,
            modelName: String = "model.onnx",
        ): SteerTtsInference {
            val options = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                // Leave a core for the audio track. 1-4 threads is the useful range for a ~12 MB
                // model on mobile; raise only if you measure a win.
                setIntraOpNumThreads(
                    (Runtime.getRuntime().availableProcessors() - 1).coerceIn(1, 4)
                )
                setInterOpNumThreads(1)
                setMemoryPatternOptimization(false) // predictable RSS on long utterances
                setCPUArenaAllocator(false)         // return activation memory after each run
            }

            val env = OrtEnvironment.getEnvironment()
            val session = try {
                // OrtSession constructors are private: sessions come from OrtEnvironment.
                env.createSession(modelBuffer, options)
            } catch (e: OrtException) {
                throw SteerTtsModelContractException(
                    "createSession failed for '$modelName': ${e.message}"
                )
            }

            val inference = SteerTtsInference(
                env = env,
                session = session,
                modelBuffer = modelBuffer,
                tokenizer = tokenizer,
                speakerEmbedding = speakerEmbedding,
                controlDim = controlDim,
                guidance = guidance,
                sampleRate = sampleRate,
                modelName = modelName,
            )
            Log.i(TAG, "Loaded ${inference.describe()}")
            return inference
        }

        private fun readJson(assets: AssetManager, path: String): JSONObject = try {
            assets.open(path).bufferedReader().use { JSONObject(it.readText()) }
        } catch (e: IOException) {
            throw SteerTtsMissingAssetException(path, e)
        }

        /**
         * ORT reads the graph out of a direct ByteBuffer, which avoids the compressed-asset trap
         * of assets.openFd(). The buffer must stay reachable while the session is alive.
         */
        private fun readModelBuffer(assets: AssetManager, path: String): ByteBuffer {
            val bytes = try {
                assets.open(path).use { it.readBytes() }
            } catch (e: IOException) {
                throw SteerTtsMissingAssetException(path, e)
            }
            if (bytes.isEmpty()) throw SteerTtsModelContractException("'$path' is empty")
            val buffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
            buffer.put(bytes)
            buffer.rewind()
            return buffer
        }

        private fun findInput(declared: Set<String>, vararg candidates: String): String? =
            candidates.firstOrNull { it in declared }

        private fun readFloats(tensor: OnnxTensor): FloatArray {
            val info = tensor.info as TensorInfo
            if (info.type == OnnxJavaType.FLOAT) {
                val buffer: FloatBuffer = tensor.floatBuffer
                return FloatArray(buffer.remaining()).also { buffer.get(it) }
            }
            // Defensive fallback for a graph that returns nested arrays.
            return flattenFloats(tensor.value)
        }

        private fun flattenFloats(value: Any?): FloatArray {
            val out = ArrayList<Float>()
            fun walk(node: Any?) {
                when (node) {
                    is FloatArray -> node.forEach { out.add(it) }
                    is Array<*> -> node.forEach { walk(it) }
                    is Float -> out.add(node)
                    is Double -> out.add(node.toFloat())
                    else -> throw SteerTtsModelContractException(
                        "Unexpected output element: ${node?.javaClass}"
                    )
                }
            }
            walk(value)
            return out.toFloatArray()
        }
    }
}

/** Creates input tensors and guarantees they are released after the run. */
private class TensorBag(private val env: OrtEnvironment) : AutoCloseable {
    private val created = ArrayList<OnnxTensor>(8)
    val map = HashMap<String, OnnxTensor>(8)

    fun putLong(name: String, data: LongArray, shape: LongArray) {
        val tensor = OnnxTensor.createTensor(env, LongBuffer.wrap(data), shape)
        created.add(tensor)
        map[name] = tensor
    }

    fun putFloat(name: String, data: FloatArray, shape: LongArray) {
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)
        created.add(tensor)
        map[name] = tensor
    }

    override fun close() {
        created.forEach { runCatching { it.close() } }
        created.clear()
        map.clear()
    }
}

/**
 * Minimal safetensors reader for the per-voice speaker vectors (fp32 tensor "s", one row per
 * voice). Layout: uint64-LE header length | JSON header | data starting at 8 + headerLength.
 */
private object Safetensors {
    fun readFloatRow(assets: AssetManager, path: String, tensor: String, row: Int): FloatArray {
        val bytes = try {
            assets.open(path).use { it.readBytes() }
        } catch (e: IOException) {
            throw SteerTtsMissingAssetException(path, e)
        }
        if (bytes.size < 8) throw SteerTtsModelContractException("'$path' is not a safetensors file")

        val headerLength = ByteBuffer.wrap(bytes, 0, 8).order(ByteOrder.LITTLE_ENDIAN).long
        if (headerLength <= 0 || headerLength > bytes.size - 8) {
            throw SteerTtsModelContractException("'$path' has a corrupt safetensors header")
        }
        val header = JSONObject(String(bytes, 8, headerLength.toInt(), Charsets.UTF_8))
        val entry = header.optJSONObject(tensor)
            ?: throw SteerTtsModelContractException("Tensor '$tensor' not found in '$path'")
        if (entry.optString("dtype") != "F32") {
            throw SteerTtsModelContractException(
                "Tensor '$tensor' is ${entry.optString("dtype")}; only F32 is supported"
            )
        }
        val shape = entry.getJSONArray("shape")
        if (shape.length() != 2) {
            throw SteerTtsModelContractException("Speaker tensor '$tensor' must be [voices, dim]")
        }
        val offsets = entry.getJSONArray("data_offsets")
        val dim = shape.getInt(1)
        val begin = offsets.getLong(0)
        val end = offsets.getLong(1)
        if (row < 0 || row >= shape.getInt(0)) {
            throw SteerTtsModelContractException(
                "speaker_index=$row is outside 0..${shape.getInt(0) - 1}"
            )
        }

        val dataStart = (8 + headerLength).toInt()
        val floatOffset = dataStart / 4 + (begin / 4).toInt() + row * dim
        val floatEnd = dataStart / 4 + (end / 4).toInt()
        if (floatOffset + dim > floatEnd) {
            throw SteerTtsModelContractException("Speaker row exceeds tensor bounds in '$path'")
        }

        val floats = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val out = FloatArray(dim)
        floats.position(floatOffset)
        floats.get(out)
        return out
    }
}

/** Float samples to 16-bit little-endian PCM, matching AudioFormat.ENCODING_PCM_16BIT. */
object Pcm16 {
    /** Short ramps at utterance edges only: mid-stream chunk cuts must stay sample-continuous. */
    private const val FADE_SAMPLES = 120 // 5 ms at 24 kHz

    fun encode(
        samples: FloatArray,
        offset: Int,
        count: Int,
        fadeIn: Boolean,
        fadeOut: Boolean,
    ): ByteArray {
        val out = ByteArray(count * 2)
        for (i in 0 until count) {
            var sample = samples[offset + i]
            if (fadeIn && i < FADE_SAMPLES) sample *= i.toFloat() / FADE_SAMPLES
            if (fadeOut && i >= count - FADE_SAMPLES) {
                sample *= (count - i).toFloat() / FADE_SAMPLES
            }
            val value = (sample.coerceIn(-1f, 1f) * 32767f).toInt()
            out[i * 2] = (value and 0xFF).toByte()
            out[i * 2 + 1] = ((value shr 8) and 0xFF).toByte()
        }
        return out
    }
}
