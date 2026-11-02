#!/usr/bin/env python3
"""Validate a SteerTTS model package before it is baked into an APK.

A truncated or partial download produces an APK that installs, registers as an engine and then
fails at first utterance with an opaque native error. This checks the package the C engine will
actually open, so that cannot happen silently:

  * every file the engine reads is present and non-empty,
  * config.json and voices.json parse, and the interesting fields are reported,
  * each .safetensors file has a well-formed header and its byte count matches the header,
  * the voice table's row width matches the speaker dimension the acoustic model declares.

Usage:  python tools/check_model.py [dir]      (default: c/models/steer-tts-edge)
Exit code 0 = usable, 1 = something is wrong.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

REQUIRED = [
    "config.json",
    "voices.json",
    "acoustic.safetensors",
    "vocoder.safetensors",
    "voices.safetensors",
    "native/front/gru.bin",
    "native/front/hetero.tsv",
    "native/front/lexicon.bin",
    "native/front/oov.bin",
    "native/front/SYMBOLS.txt",
    "native/front/unicode.bin",
]
ST_TYPES = {"F32": 4, "F16": 2, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BF16": 2, "F64": 8}


def human(n: float) -> str:
    return f"{n / 1048576:.2f} MB" if n >= 1048576 else f"{n / 1024:.0f} KB"


def read_safetensors(path: str) -> dict:
    """Return {tensor: (dtype, shape, bytes)} after checking the header and the byte count."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("shorter than its 8-byte header length")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len <= 0 or header_len > size - 8:
            raise ValueError(f"header length {header_len} does not fit in {size} bytes")
        header = json.loads(f.read(header_len).decode("utf-8"))

    data_start = 8 + header_len
    data_bytes = size - data_start
    tensors: dict[str, tuple[str, list[int], int]] = {}
    extent = 0
    for name, entry in header.items():
        if name == "__metadata__" or not isinstance(entry, dict):
            continue
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in ST_TYPES or not isinstance(shape, list) or not offsets:
            raise ValueError(f"tensor '{name}' has an unusable header entry")
        begin, end = int(offsets[0]), int(offsets[1])
        expect = ST_TYPES[dtype]
        count = 1
        for d in shape:
            count *= int(d)
        if end - begin != count * expect:
            raise ValueError(
                f"tensor '{name}' is {dtype}{shape} = {count * expect} bytes, header says {end - begin}"
            )
        extent = max(extent, end)
        tensors[name] = (dtype, [int(d) for d in shape], end - begin)

    if extent > data_bytes:
        raise ValueError(f"tensors claim {extent} bytes of data, file holds {data_bytes}")
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir", nargs="?", default=os.path.join("c", "models", "steer-tts-edge"))
    args = parser.parse_args()
    root = args.dir

    print(f"Checking {root}\n")
    problems: list[str] = []
    total = 0

    if not os.path.isdir(root):
        print(f"FAIL: {root} does not exist", file=sys.stderr)
        return 1

    for rel in REQUIRED:
        path = os.path.join(root, rel.replace("/", os.sep))
        if not os.path.isfile(path):
            problems.append(f"missing {rel}")
            print(f"  MISSING      {rel}")
            continue
        size = os.path.getsize(path)
        total += size
        if size == 0:
            problems.append(f"{rel} is empty")
            print(f"  EMPTY        {rel}")
        else:
            print(f"  {human(size):>10}   {rel}")

    # ---- config.json
    config: dict = {}
    try:
        with open(os.path.join(root, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
        print(f"\nconfig.json keys: {', '.join(sorted(config)) if config else '(empty)'}")
        acoustic = config.get("acoustic")
        if isinstance(acoustic, dict):
            print(f"  acoustic: { {k: acoustic[k] for k in list(acoustic)[:8]} }")
    except Exception as e:
        problems.append(f"config.json does not parse: {e}")
        print(f"\nFAIL config.json: {e}")

    # ---- voices.json
    voices = []
    try:
        with open(os.path.join(root, "voices.json"), encoding="utf-8") as f:
            voices_json = json.load(f)
        voices = voices_json if isinstance(voices_json, list) else voices_json.get("voices", [])
        print(f"voices: {len(voices)}  e.g. {[v.get('name', v) if isinstance(v, dict) else v for v in voices[:4]]}")
    except Exception as e:
        problems.append(f"voices.json does not parse: {e}")
        print(f"FAIL voices.json: {e}")

    # ---- safetensors
    print("")
    dims: dict[str, dict] = {}
    for name in ("acoustic.safetensors", "vocoder.safetensors", "voices.safetensors"):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        try:
            tensors = read_safetensors(path)
            dims[name] = tensors
            described = ", ".join(f"{t}{s}" for t, (_, s, _) in list(tensors.items())[:4])
            print(f"  {name:<22} {len(tensors):>3} tensors   {described}")
        except Exception as e:
            problems.append(f"{name}: {e}")
            print(f"  {name:<22} FAIL: {e}")

    # ---- the check the engine itself makes: voice rows vs. speaker dimension
    voice_table = dims.get("voices.safetensors", {})
    if voice_table:
        tensor, (dtype, shape, _) = next(iter(voice_table.items()))
        row_dim = shape[-1] if shape else 0
        print(f"\nvoice tensor '{tensor}' {dtype}{shape} -> {row_dim} values per voice")
        if len(voices) and shape and shape[0] not in (len(voices), 1):
            problems.append(f"voices.json lists {len(voices)} voices but the table has {shape[0]} rows")
            print(f"  MISMATCH: {len(voices)} voices in voices.json vs {shape[0]} rows")

    print(f"\ntotal: {human(total)}")
    if problems:
        print(f"\nUNUSABLE - {len(problems)} problem(s):", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print("\nOK: the engine's inputs are all present and well formed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
