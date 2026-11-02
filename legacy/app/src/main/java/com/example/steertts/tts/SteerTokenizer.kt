package com.example.steertts.tts

import android.content.res.AssetManager
import java.io.IOException

/**
 * STUB text front end.
 *
 * SteerTTS does not tokenize characters: it normalises text, runs a lexicon plus an int8 GRU G2P
 * to ARPAbet *phonemes*, and emits per-phone word indices. That port is the remaining work.
 * Until then this does greedy longest-match over a symbol table so tensor shapes are right and the
 * rest of the pipeline can be exercised end to end.
 *
 * `tokens.txt` must mirror the model's own symbol table: line N is id N.
 */
class SteerTokenizer private constructor(
    private val symbolToId: Map<String, Long>,
    private val unknownId: Long?,
    private val eosId: Long?,
    private val maxSymbolLength: Int,
) {
    fun encode(text: String): LongArray {
        val normalised = normalise(text)
        if (normalised.isEmpty()) return LongArray(0)

        val out = ArrayList<Long>(normalised.length + 1)
        var index = 0
        while (index < normalised.length) {
            var consumed = 0
            val maxLength = minOf(maxSymbolLength, normalised.length - index)
            for (length in maxLength downTo 1) {
                val id = symbolToId[normalised.substring(index, index + length)]
                if (id != null) {
                    out.add(id)
                    consumed = length
                    break
                }
            }
            if (consumed == 0) {
                unknownId?.let { out.add(it) }
                consumed = 1
            }
            index += consumed
        }
        eosId?.let { out.add(it) }
        return out.toLongArray()
    }

    /** Uppercase and whitespace collapse, because ARPAbet symbols are uppercase. */
    private fun normalise(text: String): String =
        text.uppercase().replace(WHITESPACE, " ").trim()

    companion object {
        private val WHITESPACE = Regex("\\s+")

        fun load(
            assets: AssetManager,
            path: String,
            unknownSymbol: String? = "<unk>",
            eosSymbol: String? = null,
        ): SteerTokenizer {
            val table = try {
                assets.open(path).bufferedReader().use { it.readText() }
            } catch (e: IOException) {
                throw SteerTtsMissingAssetException(path, e)
            }
            return fromSymbolTable(table, unknownSymbol, eosSymbol)
        }

        /** Same as [load] but from an in-memory table, so it is unit-testable without Android. */
        fun fromSymbolTable(
            table: String,
            unknownSymbol: String? = "<unk>",
            eosSymbol: String? = null,
        ): SteerTokenizer {
            val symbols = LinkedHashMap<String, Long>()
            table.lineSequence().forEachIndexed { index, raw ->
                val symbol = raw.trimEnd('\r')
                if (symbol.isNotEmpty()) symbols.putIfAbsent(symbol, index.toLong())
            }
            if (symbols.isEmpty()) throw SteerTtsModelContractException("Token table is empty")
            return SteerTokenizer(
                symbolToId = symbols,
                unknownId = unknownSymbol?.let { symbols[it] },
                eosId = eosSymbol?.let { symbols[it] },
                maxSymbolLength = symbols.keys.maxOf { it.length },
            )
        }
    }
}
