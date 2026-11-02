# SteerTTS — an Android text-to-speech engine

An offline TTS engine for Android built on **SteerTTS**, a 23M-parameter English voice you steer with
numbers: pitch, rate, loudness, pitch range, spectral tilt and seven more. It runs entirely on the
phone — nothing is uploaded.

Install it, choose it in **Settings → Accessibility → Text-to-speech output → Preferred engine**, and
every app that speaks (TalkBack, navigation, notification readers) uses this voice.

| | |
|---|---|
| Model | SteerTTS Edge, int4 — 12.5 MB |
| Voices | 200 |
| Speed | RTF ≈ 0.02 on a Pixel 10 (≈ 50× real time), ~25 ms to first audio |
| APK | 12.9 MB, `arm64-v8a`, Android 8.0+ (API 26) |
| Licence | CC BY-NC 4.0 — non-commercial |

## Install

1. Build the APK (below), or use one from `dist/`.
2. Copy it to the phone and open it; allow installation from that source.
3. Settings → Accessibility → Text-to-speech output → Preferred engine → SteerTTS.

The app's Settings tab shows whether it is already the phone's engine.

## Controls

Every change applies to the next utterance. All twelve are z-scores against the selected voice's own
neutral speech, −3…+3, where 0 leaves the decision to the model.

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
| `cpps` | cepstral peak prominence |
| `h1h2` | H1–H2 (breathiness) |

**Style presets** — Natural, Calm, Warm, Happy, Excited, Serious, Sad, Whisper. Each combines a style
tag (`whisper`, `happiness`, `serenity`, `confidence`, …) with the controls that follow it reliably.

**Speed** — 0.7–1.6. An app's own speech-rate setting scales it, and its pitch setting is converted to
semitones of `f0_med`, so TalkBack and navigation still sound natural.

**Markup** — `<whisper>quietly</whisper>`, `[pause 300ms]`, `*emphasis*`.

## Build

Requirements: JDK 17–21, Android SDK (platform 36, Build Tools 35.0.1), NDK r27b, CMake 3.22.1.

The weights are licence-gated (CC BY-NC). Accept the licence at
[huggingface.co/WhissleAI/steer-tts-edge](https://huggingface.co/WhissleAI/steer-tts-edge), create a
read token, then fetch them once:

```powershell
$env:HF_TOKEN = 'hf_...'
python tools/fetch_model.py     # -> c/models/steer-tts-edge (~12.5 MB)
```

Then:

```powershell
.\build-apk.cmd                 # -> dist/SteerTTS-release-<date>.apk
```

`tools/build-apk.ps1` accepts `-Jdk`, `-Ndk`, `-Cmake`, `-Sdk` and `-ModelDir`. It reads the signing
password from `STEERTTS_*` environment variables or from `../toolchain/signing.properties`; no
credentials are stored in this repository. `python tools/check_model.py` verifies a package before it
is packaged.

## Layout

```
android/   the app — SteerTtsService (the engine), voices and controls UI, JNI bridge
c/         the engine — acoustic model, flow, vocoder, G2P
tools/     fetch_model.py, check_model.py, build-apk.ps1
```

## Notes

- This is the **engine** build. Reading notifications aloud, announcing calls, calendar and health
  summaries and a screen reader are not included: they need notification access, `READ_CALENDAR` and
  Health Connect, which Play Protect flags on a sideloaded APK. Upstream's full assistant app is at
  [WhissleAI/steer-tts](https://github.com/WhissleAI/steer-tts).
- Only `arm64-v8a` is built. Native libraries are stored compressed, so the app also installs on
  Android 15+ devices with 16 KB memory pages.
- `c/models/`, `dist/` and the signing key are deliberately not in the repository.

## Credits and licence

Engine and app by [Whissle AI](https://github.com/WhissleAI/steer-tts). The inference code is
MIT-style; the weights and app are **CC BY-NC 4.0** — free for personal and research use, while any
commercial use needs a written licence from Whissle. Pronunciation data is derived from the CMU
Pronouncing Dictionary (BSD).
