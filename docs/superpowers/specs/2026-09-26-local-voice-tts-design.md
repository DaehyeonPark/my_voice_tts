# Local Voice TTS Design

## Goal

Create a personal Korean voice-cloning TTS web app that runs only on this Mac.
It converts user-entered text into a WAV file using the owner's supplied voice
recording. The app must not send audio, text, or generated speech to a hosted
TTS service.

## Scope

- Project root: `my_voice_tts`
- Application source: `my_voice_tts/src`
- Initial engine: F5-TTS zero-shot voice cloning
- UI: a locally served browser page on `127.0.0.1` only
- Input: Korean text and a local reference recording
- Output: playable and downloadable WAV

The provided recording at `study_src/새로운 녹음.m4a` is approximately three
minutes long. It is sufficient for the initial zero-shot implementation. More
recording is requested only if evaluation shows unstable similarity,
pronunciation, or prosody.

## Non-goals

- Hosted service, public endpoint, authentication, or multi-user accounts
- Impersonating a person whose voice was not supplied by that person
- Model fine-tuning in the initial version
- MP3 export, multiple saved voices, or mobile deployment

## Architecture

```text
Browser (127.0.0.1)
  -> FastAPI routes
    -> request validation and text chunking
      -> reference-audio preparation (FFmpeg)
      -> F5-TTS inference adapter
      -> WAV concatenation and output store
  <- status, audio stream, and download URL
```

The backend is split into focused modules:

- `app.py`: FastAPI application and local-only server configuration.
- `api.py`: HTTP request and response contracts.
- `services/audio.py`: audio inspection, conversion to 24 kHz mono WAV, and
  generated WAV concatenation.
- `services/text.py`: Korean text validation and sentence-aware chunking.
- `services/tts.py`: F5-TTS model loading and inference through an interface
  that permits a future engine replacement.
- `services/jobs.py`: output creation, names, and cleanup policy.
- `static/`: single-page local web UI.

Runtime content is separate from source:

- `study_src/`: user-owned original recordings; never modified by the app.
- `data/reference/`: prepared reference WAV and its user-confirmed transcript.
- `data/outputs/`: generated WAV files.
- `data/models/`: local model cache when configuration supports it.

The audio and runtime data directories are ignored by Git. Only a placeholder
file is tracked where required to describe the directory layout.

## User flow

1. The user starts the app with the documented local command and opens the
   loopback URL in a browser.
2. On first use, the app offers the supplied recording as a reference source.
   It converts a chosen clean 10--20 second region to a model-ready WAV.
3. The user enters or confirms the exact transcript for that reference region.
   The app stores that transcript beside the prepared reference audio.
4. The user enters Korean text and clicks **Generate**.
5. The backend rejects empty or excessively long requests, splits valid text
   at sentence boundaries, synthesizes each chunk, and joins the results.
6. The page displays an audio player and a download control for the generated
   WAV. Output stays on the local disk until deleted by the user or the cleanup
   policy.

## Privacy and safety

- The server binds exclusively to `127.0.0.1`; no LAN binding or share link is
  enabled.
- No telemetry, cloud TTS call, upload endpoint, or remote storage is used.
- First-run model retrieval is the sole network operation and is visibly
  reported. Once downloaded, inference runs offline.
- The app identifies this as a personal voice tool and requires the user to
  confirm they have permission to use the reference voice.
- File paths are generated server-side; client input cannot select arbitrary
  output paths.

## Failure handling

- Missing FFmpeg, model dependencies, or model files produce an actionable
  setup error rather than a failed browser request.
- Unsupported/corrupt reference audio is rejected before inference.
- A missing reference transcript blocks synthesis and explains why it matters.
- A failed chunk stops the job, removes temporary partial output, and reports
  the failing sentence number.
- Model loading is serialized so simultaneous browser actions do not load the
  model twice.

## Testing and acceptance

Automated tests cover:

- text normalization, blank-input rejection, and sentence chunking;
- safe output-name and reference-path handling;
- audio preparation command construction and failure mapping;
- API validation and loopback-only configuration;
- orchestration behavior with a fake TTS engine, including cleanup on failed
  chunks.

An optional manual smoke command performs real F5-TTS inference with the
prepared reference and verifies that the output is a non-empty 24 kHz WAV.

Acceptance criteria:

1. The app runs locally on Apple Silicon and does not listen beyond loopback.
2. A Korean text request generates a playable WAV using the supplied reference
   voice.
3. The result can be downloaded from the local UI.
4. The app gives clear guidance if additional, cleaner recording is needed.
