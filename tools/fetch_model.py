#!/usr/bin/env python3
"""Download the SteerTTS edge model package into a local directory.

The weights are licence-gated on Hugging Face (CC BY-NC 4.0). Before running this:

  1. Open https://huggingface.co/WhissleAI/steer-tts-edge and accept the licence
     (log in, then click through the access conditions).
  2. Create a READ token at https://huggingface.co/settings/tokens .
  3. Export it as HF_TOKEN (do not commit it):

         setx HF_TOKEN hf_xxxxxxxxxxxxxxxxxxxxx       # Windows, new shells only
         export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx     # bash / zsh

Then:

    python tools/fetch_model.py                      # -> c/models/steer-tts-edge
    python tools/fetch_model.py --dest my-model-dir

The downloader uses Python's own TLS stack on purpose: on some Windows machines the
system schannel backend rejects the handshake (SEC_E_NO_CREDENTIALS), which breaks both
curl.exe and PowerShell's Invoke-WebRequest for huggingface.co. Python's bundled OpenSSL
does not have that problem.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.error
import urllib.request

REPO = "WhissleAI/steer-tts-edge"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"

# Everything the C engine opens, from a fresh config.json read in st_api.c.
REQUIRED = [
    "config.json",
    "voices.json",
    "acoustic.safetensors",
    "vocoder.safetensors",
    "voices.safetensors",
    # text front end: lexicon + int8 GRU G2P + unicode tables
    "native/front/gru.bin",
    "native/front/hetero.tsv",
    "native/front/lexicon.bin",
    "native/front/oov.bin",
    "native/front/SYMBOLS.txt",
    "native/front/unicode.bin",
]

# Optional: only needed for --perform / the director styles.
OPTIONAL = ["director.bin"]


def human(n: int) -> str:
    return f"{n / 1048576:.1f} MB"


def fetch(name: str, dest_dir: str, token: str) -> bool:
    """Download one file. Returns False for a missing optional file."""
    target = os.path.join(dest_dir, name.replace("/", os.sep))
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    request = urllib.request.Request(
        BASE + name,
        headers={"Authorization": f"Bearer {token}", "User-Agent": "steertts-fetch/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as out:
            expected = int(response.headers.get("Content-Length") or 0)
            shutil.copyfileobj(response, out, 1 << 20)
            got = os.path.getsize(target)
            flag = "" if not expected or got == expected else f"  (expected {human(expected)})"
            print(f"  {name:<32} {human(got)}{flag}")
        return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            print(
                f"\nHTTP {e.code} for {name}: the token is not authorised.\n"
                f"Accept the licence at https://huggingface.co/{REPO} first, then use a READ token\n"
                "from https://huggingface.co/settings/tokens .",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if e.code == 404 and name in OPTIONAL:
            print(f"  {name:<32} not published (optional, skipped)")
            return False
        raise
    except urllib.error.URLError as e:
        print(f"\nfailed to reach huggingface.co: {e.reason}", file=sys.stderr)
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default=os.path.join("c", "models", "steer-tts-edge"),
                        help="destination directory (default: c/models/steer-tts-edge)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face read token (default: $HF_TOKEN)")
    args = parser.parse_args()

    if not args.token:
        print("No token: set HF_TOKEN or pass --token.\n"
              f"Accept the licence first at https://huggingface.co/{REPO}", file=sys.stderr)
        return 2

    print(f"{REPO} -> {args.dest}")
    for name in REQUIRED:
        fetch(name, args.dest, args.token)
    for name in OPTIONAL:
        fetch(name, args.dest, args.token)

    print("\nok: pass this directory to Gradle with  -PsteerttsModel=<dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
