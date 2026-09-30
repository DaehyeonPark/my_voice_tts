# Local Voice TTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build a loopback-only Korean voice-cloning TTS web app that uses the owner's local recording and F5-TTS to create downloadable WAV files.

**Architecture:** A FastAPI server on 127.0.0.1 serves a small static UI and narrow JSON API. Pure services handle text, audio preparation, reference metadata, and job orchestration; an F5 adapter implements the inference protocol and is lazy-loaded so automated tests use a fake engine.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, Pydantic, F5-TTS, FFmpeg, pytest, vanilla HTML/CSS/JavaScript.

**Spec:** my_voice_tts/docs/superpowers/specs/2026-09-26-local-voice-tts-design.md

## Global Constraints

- Source lives in my_voice_tts/src; tests live in my_voice_tts/tests.
- Bind only to 127.0.0.1; do not expose a LAN address or share link.
- Keep study_src/ untouched; never commit source recordings, model files, prepared audio, or outputs.
- Use F5-TTS F5TTS_v1_Base through its Python API; it may download model files only at first model load, then runs locally.
- Prepare reference audio as 24 kHz mono WAV and require a non-blank exact transcript.
- Generate WAV only; do not add cloud TTS, telemetry, MP3, accounts, or fine-tuning.
- On a multi-sentence failure, remove all temporary audio and return the one-based failing sentence number.

## Review Focus

- A path-traversal source or output ID must fail without escaping project directories (Task 5).
- A whitespace-only reference transcript must block preparation and synthesis with a readable 422 response (Tasks 3 and 5).
- An FFmpeg process failure or missing executable must be an actionable setup error, not a traceback (Task 3).
- A single sentence longer than the configured maximum must return 422 rather than silently truncate (Task 2).
- A later chunk failure must leave no partial WAV or temporary chunks (Task 4).

---

## File Structure

    my_voice_tts/
      pyproject.toml                 package metadata and dependencies
      .gitignore                     protects recordings, cache, and outputs
      README.md                      setup, privacy, and run instructions
      src/voice_tts/
        config.py                    typed paths and fixed settings
        schemas.py                   API Pydantic models
        app.py                       factory and loopback entry point
        api.py                       HTTP routes and error translation
        services/text.py             input validation and sentence chunking
        services/audio.py            FFmpeg preparation and WAV joining
        services/reference.py        source discovery and metadata store
        services/tts.py              protocol and lazy F5-TTS adapter
        services/jobs.py             serialized synthesis and cleanup
        static/index.html
        static/app.js
        static/styles.css
      tests/
        conftest.py
        test_config.py
        test_text.py
        test_audio.py
        test_reference.py
        test_tts.py
        test_jobs.py
        test_api.py

### Task 1: Bootstrap configuration and safe local directories

**Files:**
- Create: my_voice_tts/pyproject.toml
- Create: my_voice_tts/.gitignore
- Create: my_voice_tts/src/voice_tts/__init__.py
- Create: my_voice_tts/src/voice_tts/config.py
- Create: my_voice_tts/tests/test_config.py

**Interfaces:**
- Produces: ProjectPaths.from_root(root: Path) -> ProjectPaths; ProjectPaths.ensure_runtime_directories() -> None; Settings(host: str = "127.0.0.1", port: int = 8765, reference_sample_rate: int = 24000, reference_channels: int = 1, max_chunk_chars: int = 260).
- Consumed by: all later services and the application factory.

- [ ] **Step 1: Write failing configuration tests**

    def test_runtime_creation_does_not_create_or_modify_study_source(tmp_path):
        paths = ProjectPaths.from_root(tmp_path)
        paths.ensure_runtime_directories()
        assert paths.reference_dir.is_dir()
        assert paths.output_dir.is_dir()
        assert not paths.study_source_dir.exists()

    def test_settings_are_loopback_only():
        assert Settings().host == "127.0.0.1"

- [ ] **Step 2: Run the configuration tests to verify they fail**

Run: cd my_voice_tts && pytest tests/test_config.py -v

Expected: FAIL because voice_tts.config does not exist.

- [ ] **Step 3: Add package metadata and ProjectPaths**

Create Python 3.11 package metadata with FastAPI, Uvicorn, F5-TTS, and pytest. ProjectPaths owns study_src, data/reference, data/outputs, data/models, and tmp. It creates only runtime paths. Ignore .venv/, data/, tmp/, study_src/, *.wav, and *.m4a.

- [ ] **Step 4: Run the configuration tests to verify they pass**

Run: cd my_voice_tts && pytest tests/test_config.py -v

Expected: PASS.

- [ ] **Step 5: Commit**

Run: git add my_voice_tts/pyproject.toml my_voice_tts/.gitignore my_voice_tts/src/voice_tts my_voice_tts/tests/test_config.py && git commit -m "feat: bootstrap local voice tts app"

### Task 2: Validate and split Korean synthesis text

**Files:**
- Create: my_voice_tts/src/voice_tts/services/__init__.py
- Create: my_voice_tts/src/voice_tts/services/text.py
- Create: my_voice_tts/tests/test_text.py

**Interfaces:**
- Consumes: Settings.max_chunk_chars.
- Produces: validate_and_chunk_text(text: str, max_chars: int) -> list[str]; TextValidationError(ValueError).
- Consumed by: SynthesisService in Task 4.

- [ ] **Step 1: Write failing text-service tests**

    def test_splits_at_korean_sentence_boundaries():
        assert validate_and_chunk_text("첫 문장입니다. 둘째 문장입니다.", 12) == ["첫 문장입니다.", "둘째 문장입니다."]

    def test_rejects_blank_and_single_oversized_sentence():
        with pytest.raises(TextValidationError, match="텍스트를 입력"):
            validate_and_chunk_text("  ", 20)
        with pytest.raises(TextValidationError, match="한 문장이 너무 깁니다"):
            validate_and_chunk_text("가" * 21, 20)

- [ ] **Step 2: Run the text tests to verify they fail**

Run: cd my_voice_tts && pytest tests/test_text.py -v

Expected: FAIL because validate_and_chunk_text is undefined.

- [ ] **Step 3: Implement validation and sentence-aware chunking**

Implement validate_and_chunk_text(text: str, max_chars: int) -> list[str]. Normalize whitespace, preserve terminal punctuation (. ! ? 。 ！ ？), pack only whole sentences up to the limit, and do not split a single oversized sentence.

- [ ] **Step 4: Run the text tests to verify they pass**

Run: cd my_voice_tts && pytest tests/test_text.py -v

Expected: PASS.

- [ ] **Step 5: Commit**

Run: git add my_voice_tts/src/voice_tts/services/text.py my_voice_tts/tests/test_text.py && git commit -m "feat: validate and split synthesis text"

### Task 3: Prepare reference audio and store confirmed transcript

**Files:**
- Create: my_voice_tts/src/voice_tts/services/audio.py
- Create: my_voice_tts/src/voice_tts/services/reference.py
- Create: my_voice_tts/tests/test_audio.py
- Create: my_voice_tts/tests/test_reference.py

**Interfaces:**
- Consumes: ProjectPaths and reference sample settings.
- Produces: AudioPreparationError; prepare_reference(source: Path, start_seconds: float, duration_seconds: float, destination: Path) -> None; concatenate_wavs(inputs: list[Path], destination: Path) -> None; Reference(path: Path, transcript: str); ReferenceStore.prepare(source_name: str, start_seconds: float, duration_seconds: float, transcript: str) -> Reference.
- Consumed by: Task 4 and Task 5.

- [ ] **Step 1: Write failing audio and reference tests**

    def test_prepare_reference_requests_24k_mono_wav(tmp_path, runner):
        prepare_reference(Path("input.m4a"), 0, 15, tmp_path / "reference.wav", runner)
        assert runner.calls[0][:4] == ["ffmpeg", "-y", "-ss", "0"]

    def test_ffmpeg_failure_is_actionable(tmp_path, failing_runner):
        with pytest.raises(AudioPreparationError, match="FFmpeg"):
            prepare_reference(Path("input.m4a"), 0, 15, tmp_path / "reference.wav", failing_runner)

    def test_reference_store_rejects_blank_transcript(project_paths):
        with pytest.raises(ValueError, match="대본"):
            ReferenceStore(project_paths, fake_audio).prepare("voice.m4a", 0, 15, " ")

- [ ] **Step 2: Run the audio and reference tests to verify they fail**

Run: cd my_voice_tts && pytest tests/test_audio.py tests/test_reference.py -v

Expected: FAIL because the services do not exist.

- [ ] **Step 3: Implement audio conversion and reference metadata**

Use FFmpeg with an explicit input seek, duration, -ac 1, -ar 24000, and WAV output. Map missing FFmpeg and nonzero process results to AudioPreparationError. Resolve source_name beneath study_src only, then write reference.wav and reference.json under data/reference; reject any escaping name.

- [ ] **Step 4: Run the audio and reference tests to verify they pass**

Run: cd my_voice_tts && pytest tests/test_audio.py tests/test_reference.py -v

Expected: PASS.

- [ ] **Step 5: Commit**

Run: git add my_voice_tts/src/voice_tts/services/audio.py my_voice_tts/src/voice_tts/services/reference.py my_voice_tts/tests/test_audio.py my_voice_tts/tests/test_reference.py && git commit -m "feat: prepare local voice reference audio"

### Task 4: Add F5-TTS adapter and cleanup-safe synthesis

**Files:**
- Create: my_voice_tts/src/voice_tts/services/tts.py
- Create: my_voice_tts/src/voice_tts/services/jobs.py
- Create: my_voice_tts/tests/conftest.py
- Create: my_voice_tts/tests/test_tts.py
- Create: my_voice_tts/tests/test_jobs.py

**Interfaces:**
- Consumes: Reference, validate_and_chunk_text, and ProjectPaths.output_dir.
- Produces: TtsEngine(Protocol).synthesize(reference_audio: Path, reference_text: str, text: str, output_path: Path) -> None; F5TtsEngine; GeneratedAudio(output_id: str, path: Path); SynthesisService.generate(text: str) -> GeneratedAudio; SynthesisError.
- Consumed by: Task 5.

- [ ] **Step 1: Write failing adapter and job tests**

    def test_f5_adapter_loads_once_and_writes_requested_file(tmp_path, monkeypatch):
        engine = F5TtsEngine(cache_dir=tmp_path)
        engine.synthesize(Path("ref.wav"), "참조 대본", "생성 문장", tmp_path / "out.wav")
        assert fake_f5.instances == 1
        assert fake_f5.calls[0]["file_wave"].endswith("out.wav")

    def test_failed_later_chunk_removes_partial_audio(project_paths, fake_engine):
        service = SynthesisService(project_paths, prepared_reference, fake_engine, max_chunk_chars=12)
        with pytest.raises(SynthesisError, match="2번째 문장"):
            service.generate("첫 문장입니다. 실패 문장입니다.")
        assert list(project_paths.output_dir.glob("*.wav")) == []

- [ ] **Step 2: Run the adapter and job tests to verify they fail**

Run: cd my_voice_tts && pytest tests/test_tts.py tests/test_jobs.py -v

Expected: FAIL because the adapter and job service do not exist.

- [ ] **Step 3: Implement the engine protocol and F5 adapter**

Implement a lazy, lock-protected F5TtsEngine. On first synthesis import f5_tts.api.F5TTS and construct F5TTS(model="F5TTS_v1_Base", hf_cache_dir=str(cache_dir)); call infer(ref_file=..., ref_text=..., gen_text=..., file_wave=...). Do not force a device: F5-TTS chooses mps on this Apple Silicon Mac. Translate initialization and inference exceptions to SynthesisError.

- [ ] **Step 4: Implement chunk orchestration**

SynthesisService.generate(text: str) -> GeneratedAudio loads the prepared reference, validates and chunks text, synthesizes in a job temp directory, calls concatenate_wavs after every chunk succeeds, then atomically moves one UUID WAV to data/outputs. On failure remove the temporary directory and include the one-based chunk number in SynthesisError.

- [ ] **Step 5: Run the adapter and job tests to verify they pass**

Run: cd my_voice_tts && pytest tests/test_tts.py tests/test_jobs.py -v

Expected: PASS.

- [ ] **Step 6: Commit**

Run: git add my_voice_tts/src/voice_tts/services/tts.py my_voice_tts/src/voice_tts/services/jobs.py my_voice_tts/tests/conftest.py my_voice_tts/tests/test_tts.py my_voice_tts/tests/test_jobs.py && git commit -m "feat: generate speech through local f5 tts"

### Task 5: Expose loopback API and single-page UI

**Files:**
- Create: my_voice_tts/src/voice_tts/schemas.py
- Create: my_voice_tts/src/voice_tts/api.py
- Create: my_voice_tts/src/voice_tts/app.py
- Create: my_voice_tts/src/voice_tts/static/index.html
- Create: my_voice_tts/src/voice_tts/static/app.js
- Create: my_voice_tts/src/voice_tts/static/styles.css
- Create: my_voice_tts/tests/test_api.py

**Interfaces:**
- Consumes: ReferenceStore, SynthesisService, Settings, ProjectPaths, and TtsEngine.
- Produces: create_app(paths: ProjectPaths, settings: Settings, engine: TtsEngine | None = None) -> FastAPI; GET /; GET /api/reference/sources; POST /api/reference/prepare; GET /api/reference; POST /api/synthesis; GET /api/outputs/{output_id}.

- [ ] **Step 1: Write failing API and page contract tests**

    def test_prepare_synthesis_and_download_contract(client):
        prepared = client.post("/api/reference/prepare", json={"source_name": "voice.m4a", "start_seconds": 0, "duration_seconds": 15, "transcript": "안녕하세요."})
        assert prepared.status_code == 200
        generated = client.post("/api/synthesis", json={"text": "테스트 문장입니다."})
        assert generated.status_code == 200
        assert generated.json()["download_url"].startswith("/api/outputs/")

    def test_rejects_path_traversal_and_blank_transcript(client):
        assert client.post("/api/reference/prepare", json={"source_name": "../secret.m4a", "start_seconds": 0, "duration_seconds": 15, "transcript": "대본"}).status_code == 404
        assert client.post("/api/reference/prepare", json={"source_name": "voice.m4a", "start_seconds": 0, "duration_seconds": 15, "transcript": " "}).status_code == 422

    def test_home_page_exposes_local_voice_workflow(client):
        response = client.get("/")
        assert response.status_code == 200
        assert "참조 음성" in response.text
        assert "WAV 다운로드" in response.text
        assert "127.0.0.1에서만" in response.text

- [ ] **Step 2: Run API tests to verify they fail**

Run: cd my_voice_tts && pytest tests/test_api.py -v

Expected: FAIL because create_app and routes do not exist.

- [ ] **Step 3: Implement schemas, routes, factory, and UI**

Use Pydantic models with start_seconds >= 0 and duration_seconds from 10 through 20. Return Korean 422 details for validation, 404 for absent source/output paths, and 503 for preparation or model errors. Serve static files, bind Uvicorn only to 127.0.0.1:8765, and do not accept a host override. The Korean page must select a recording, prepare a 10--20 second reference with exact transcript, submit text, disable controls while working, show errors, play the result, and provide a WAV download. Its JavaScript may call only relative /api URLs and must state that audio/text remain local after first model download.

- [ ] **Step 4: Run API tests to verify they pass**

Run: cd my_voice_tts && pytest tests/test_api.py -v

Expected: PASS.

- [ ] **Step 5: Commit**

Run: git add my_voice_tts/src/voice_tts/schemas.py my_voice_tts/src/voice_tts/api.py my_voice_tts/src/voice_tts/app.py my_voice_tts/src/voice_tts/static my_voice_tts/tests/test_api.py && git commit -m "feat: add local voice tts web app"

### Task 6: Document setup and smoke-test real local inference

**Files:**
- Create: my_voice_tts/scripts/smoke.py
- Create: my_voice_tts/README.md
- Modify: my_voice_tts/pyproject.toml
- Modify: my_voice_tts/tests/test_config.py

**Interfaces:**
- Consumes: ProjectPaths, ReferenceStore, and SynthesisService.
- Produces: voice-tts console entry point and python scripts/smoke.py.

- [ ] **Step 1: Write failing entry-point test**

    def test_project_declares_local_run_entry_point():
        metadata = Path("pyproject.toml").read_text()
        assert "voice-tts" in metadata
        assert "voice_tts.app:main" in metadata

- [ ] **Step 2: Run the test to verify it fails**

Run: cd my_voice_tts && pytest tests/test_config.py::test_project_declares_local_run_entry_point -v

Expected: FAIL because the console entry point is absent.

- [ ] **Step 3: Add run entry point, smoke script, and README**

Declare voice-tts = "voice_tts.app:main". The smoke script must refuse missing prepared reference, synthesize one short Korean test sentence through SynthesisService, and verify a non-empty WAV. README must state venv setup, FFmpeg installation, dependency install, one-time model download, voice-tts command, loopback URL, transcript requirement, output location, troubleshooting, and that study_src remains unchanged.

- [ ] **Step 4: Run full verification**

Run: cd my_voice_tts && pytest -v && python -m compileall src scripts && python scripts/smoke.py

Expected: automated tests and Python compilation PASS. Smoke either produces a non-empty 24 kHz WAV or clearly reports the one-time missing FFmpeg/model/reference prerequisite.

- [ ] **Step 5: Commit**

Run: git add my_voice_tts/pyproject.toml my_voice_tts/scripts/smoke.py my_voice_tts/README.md my_voice_tts/tests/test_config.py && git commit -m "docs: add local voice tts setup guide"

## Plan Self-Review

- Spec coverage: Tasks 1--6 cover local-only hosting, protected data, conversion and transcript, F5 inference, sentence chunking, WAV output/download, failure cleanup, UI, tests, and smoke verification.
- Type consistency: ProjectPaths, Reference, TtsEngine, and GeneratedAudio are introduced before consuming tasks.
- Review Focus: Every listed failure mode is named in a test in its owning task.
- Scope: The plan excludes fine-tuning, cloud services, MP3, multi-voice management, and public hosting.
