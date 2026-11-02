package com.example.steertts.tts

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class SteerTokenizerTest {

    @Test
    fun `ids follow the symbol table order`() {
        val tokenizer = SteerTokenizer.fromSymbolTable("AA0\nAA1\nB\nK\n")
        assertArrayEquals(longArrayOf(2, 3), tokenizer.encode("BK"))
    }

    @Test
    fun `longest match wins`() {
        val tokenizer = SteerTokenizer.fromSymbolTable("A\nAB\nB\n")
        // "AB" must match the 2-character symbol (id 1), not A + B.
        assertArrayEquals(longArrayOf(1), tokenizer.encode("AB"))
    }

    @Test
    fun `input is normalised to uppercase and whitespace collapsed`() {
        // Table is AA0=0, B=1, K=2, so "  bk  " must encode as B then K.
        val tokenizer = SteerTokenizer.fromSymbolTable("AA0\nB\nK\n")
        assertArrayEquals(longArrayOf(1, 2), tokenizer.encode("  bk  "))
    }

    @Test
    fun `unknown symbols are skipped when no unk id exists`() {
        val tokenizer = SteerTokenizer.fromSymbolTable("AA0\nB\n")
        assertArrayEquals(longArrayOf(1), tokenizer.encode("B?"))
    }

    @Test
    fun `unknown symbols map to unk when present`() {
        val tokenizer = SteerTokenizer.fromSymbolTable("AA0\nB\n<unk>\n")
        assertArrayEquals(longArrayOf(1, 2), tokenizer.encode("B?"))
    }

    @Test
    fun `eos is appended when configured`() {
        val tokenizer = SteerTokenizer.fromSymbolTable("AA0\nB\n<eos>\n", eosSymbol = "<eos>")
        assertArrayEquals(longArrayOf(1, 2), tokenizer.encode("B"))
    }

    @Test
    fun `blank input produces an empty sequence`() {
        val tokenizer = SteerTokenizer.fromSymbolTable("AA0\n")
        assertEquals(0, tokenizer.encode("   ").size)
    }
}
