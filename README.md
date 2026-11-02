# SteerTTS — an Android text-to-speech engine

An offline, on-device TTS engine for Android, built around **SteerTTS**: a 23M-parameter English
voice you steer with numbers (pitch, rate, loudness, pitch range, spectral tilt, HNR and six more),
by the [Whissle AI](https://github.com/WhissleAI/steer-tts) team.

Install it, pick it in **Settings → Accessibility → Text-to-speech output → Preferred engine**, and
every app that speaks (TalkBack, navigation, notification readers) uses this voice instead of Google's.

|  |  |
|---|---|
| Model | SteerTTS Edge — int4, 12.5 MB of weights, ~25 ms to first audio |
| Voices | 200 (female/male defaults, each with its own neutral delivery) |
| Speed | RTF ≈ 0.017–0.020 on a Pixel 10 (≈ 50× faster than real time) |
| APK | 12.9 MB, `arm64-v8a` only, Android 8.0+ (API 26) |
| Network | none — the model runs on the phone, nothing is uploaded |
| Licence | weights **CC BY-NC 4.0, non-commercial only** (see below) |

## Status

The app builds and installs today. The engine is the upstream C library (hand-written kernels for
the acoustic model, flow and vocoder — no ONNX Runtime, no Python), compiled into the APK's JNI
library by the NDK, so it is the same code that runs on the desktop and the Raspberry Pi port.

What is *not* included in this repository: **the weights**. They are licence-gated on Hugging Face,
so each person fetches them with their own account (below). Without them the APK installs and
registers as an engine but produces no audio.

What this build was checked against, mechanically:

* `tools/check_model.py` passes on the package: 11 files, 200 voices, `acoustic.safetensors`
  720 tensors, `vocoder.safetensors` 107 tensors, voice table `s[200,64]` F16 with
  `spk_mean[200,12]` — the per-voice means of the 12 controls.
* the 11 model assets inside the APK are **byte-identical** (SHA-256) to the downloaded package;
* `libsteertts_jni.so` exports all 11 `Java_ai_whissle_steertts_demo_SteerTTS_*` entry points the
  Java side declares;
* the APK is signed (v1/v2) and `arm64-v8a` only.

Not checked here: that it *sounds* right. That needs a phone — this machine has no arm64 device
and no host C compiler to render a reference WAV.

## Why this does not build the original Kotlin app

The first version of this project (kept in [`legacy/`](legacy) for reference) was written from a
specification that assumed the model was **one ONNX file** (`assets/model.onnx`) with twelve float
inputs, run through ONNX Runtime Mobile. That assumption is wrong, and it is why that code cannot
produce speech:

* the weights ship as **safetensors** (`acoustic.safetensors`, `vocoder.safetensors`,
  `voices.safetensors`), not ONNX;
* inference is a **hand-written C engine** (`c/src/st_acoustic.c`, `st_flow.c`, `st_vocoder.c`)
  that also does the mel front end, duration rounding, length regulation and the inverse STFT that
  a naive `enc/dec/voc` graph export would leave to the caller;
* the text front end is a **lexicon + int8 GRU G2P** (`c/src/st_wordpiece.c`, phonetic dictionary),
  not a character table — `legacy/` stubs it out, so even with weights it would mispronounce.

Rewriting that in Kotlin/ONNX Runtime would mean reimplementing ~10k lines of numeric code with no
way to verify it. Using the upstream engine instead gives working audio, the real 12 controls, and
the 200 bundled voices.

## Get the model (one-time, per person)

The weights are **CC BY-NC 4.0**: free for personal and research use, not for a commercial product
or a paid/hosted service. Commercial use needs a written licence from Whissle
(`ksingla@whissle.ai`).

1. Accept the licence: <https://huggingface.co/WhissleAI/steer-tts-edge>
2. Create a **read** token: <https://huggingface.co/settings/tokens>
3. Fetch the package (~13 MB):

```powershell
$env:HF_TOKEN = 'hf_...'          # for this session only; do not commit it
python tools/fetch_model.py       # -> c/models/steer-tts-edge
```

`tools/fetch_model.py` uses Python's TLS stack deliberately: on some Windows machines the system
backend rejects the handshake with `SEC_E_NO_CREDENTIALS`, which breaks `curl.exe` and
`Invoke-WebRequest` for huggingface.co.

## Build

Requirements (already installed on the machine this was set up on, under `../toolchain/`):

* **JDK 17–21** (not 25 — Gradle 8.13 will not run on it)
* **Android SDK** with platform 36 and Build Tools 35.0.1
* **NDK r27b** (`27.1.12297006`)
* **CMake 3.22.1**

Then, from the repository root:

```powershell
.\build-apk.cmd
```

or, on any platform:

```bash
cd android && ./gradlew assembleRelease
```

Both produce `dist/SteerTTS-release-<date>.apk`. `tools/build-apk.ps1` is the explicit version: it
resolves the JDK/NDK/CMake/SDK (override with `-Jdk`, `-Ndk`, `-Cmake`, `-Sdk`), writes the
machine-specific `android/local.properties`, signs with the key in `../toolchain/`, and copies the
APK to `dist/`.

Notes on the build:

* Only `arm64-v8a` is built: the APK is for real phones, not emulators.
* Native libraries are stored compressed and extracted at install, so the app also installs on
  Android 15+ devices with 16 KB memory pages.
* Keep `../toolchain/steertts-release.keystore` safe. Android only accepts an update signed with
  the same key, so losing it means uninstalling (and losing settings) before the next install.

## Install on the phone

1. Copy `dist/SteerTTS-release-*.apk` to the phone (USB, Drive, WhatsApp — any way).
2. Open it and allow installation from that source when asked.
3. **Settings → Accessibility → Text-to-speech output → Preferred engine → SteerTTS.**
4. Tap the gear next to it (that opens the app) to choose the voice and style.

The app's Settings tab shows whether it is already the phone's engine, with a shortcut to the
system screen.

## Controls

Everything is delivered per utterance, so a change applies to the next thing the phone says.

**Voices** — 200 bundled voices (female and male defaults), each with its own neutral speech; the
12 controls are z-scores *relative to the selected voice*.

**12 acoustic controls** (`VoicesScreen`, one slider each, −3…+3, 0 = leave it to the model):

| Control | What it moves |
|---|---|
| `rate` | articulation rate |
| `loud` | speech level |
| `f0_med` | median pitch |
| `f0_iqr` | pitch range |
| `f0_end` | final pitch slope (falls = statement, rises = question) |
| `pause` | pause share |
| `tilt` | spectral tilt (breathy ↔ pressed) |
| `voiced` | voiced share |
| `flat` | spectral flatness |
| `centroid` | spectral centroid (brightness) |
| `cpps` | cepstral peak prominence (voice quality) |
| `h1h2` | H1–H2 (breathiness) |

**Style tags** — `whisper`, `happiness`, `sadness`, `excited`, `serenity`, `confidence`,
`contentment`, … set the delivery the model understands best; the presets **Natural, Calm, Warm,
Happy, Excited, Serious, Sad, Whisper** are one tap each and combine a tag with the controls that
follow it reliably.

**Speed** — a separate pace multiplier (0.7–1.6). A calling app's own speech-rate setting scales
it, and its pitch setting is converted to `f0_med` in semitones.

**Inline markup** in the text — `*emphasis*`, `[pause 300ms]`, `<whisper>like this</whisper>`.

**System-wide behaviour** — `ContextStyler` adjusts delivery to context (a call is announced as an
alert, late at night softer); the engine runs in its own `:tts` process and is only held open while
in use.

## Layout

```
android/            the app (Gradle project; `app` module is the engine + UI)
  app/src/main/cpp/ JNI bridge; CMake builds ../../c into libsteertts_jni.so
  app/src/main/java/ai/whissle/steertts/demo/
                    SteerTtsService.java  the TTS engine
                    SteerTTS.java         engine handle + model install
                    Style.java            12 controls, tags, presets, persistence
                    VoicesScreen.java     the sliders and voice picker
c/                  the engine: acoustic model, flow, vocoder, G2P, WAV/MP3 out, CLI
legacy/             the original Kotlin/ONNX scaffold (superseded; see above)
tools/fetch_model.py    download the gated weights
tools/build-apk.ps1     one-command build
dist/               built APKs
```

## Credits and licence

* Engine and app: Whissle AI, [WhissleAI/steer-tts](https://github.com/WhissleAI/steer-tts) —
  inference code under the repository's MIT-style terms, app and weights under **CC BY-NC 4.0**.
* Pronunciation data derived from the CMU Pronouncing Dictionary (BSD).
* This repository adds the build setup, the model fetcher and this documentation.

The weights are non-commercial. Copies, re-hosts or forks of this app must carry the same
restriction, and generated audio may not be used commercially without a licence from Whissle.
