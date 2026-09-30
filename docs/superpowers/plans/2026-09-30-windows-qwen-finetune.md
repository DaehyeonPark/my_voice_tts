# Windows Qwen 음성 파인튜닝 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Windows RTX 3070에서 사용자의 한국어 녹음으로 Qwen3-TTS 0.6B Base를 파인튜닝하고, 검증된 모델만 Mac 로컬 웹앱에서 사용한다.

**Architecture:** Windows/WSL에서 녹음을 VAD·ASR 발화 후보로 나누고 대본과 검토 매니페스트로 정제한다. 승인된 JSONL만 학습한다. 고정 홀드아웃 평가를 통과한 모델만 활성화하며, Mac 앱은 유효한 모델이 없으면 현재의 Qwen zero-shot 복제를 유지한다.

**Tech Stack:** Python 3.11, FastAPI, Qwen3-TTS 0.6B Base, PyTorch CUDA, WSL2, FFmpeg, faster-whisper, pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-windows-qwen-finetune-design.md`

## Global Constraints

- 오디오·모델은 사용자 소유 Mac/Windows PC에만 저장한다. 클라우드 업로드는 하지 않는다.
- 골프 잡담과 경계에 겹치는 모든 발화 `[289.0, 311.0]`초는 학습과 검증에서 제외한다.
- 모델은 `Qwen/Qwen3-TTS-12Hz-0.6B-Base`로 고정한다. RTX 3070 8GB 초기값은 batch 1, gradient accumulation 8, 3 epochs다.
- 대본 불일치 또는 미승인 발화는 JSONL에 포함하지 않는다.
- OOM/검증 실패 시 active 모델을 만들지 않는다. 앱은 zero-shot으로 계속 동작해야 한다.

## Review Focus

- 제외 시간대의 경계 겹침이 모두 거절되는가.
- 학습 텍스트가 사용자가 읽은 대본과 정확히 대응하는가.
- 평가 전에는 활성 모델을 변경하지 않는가.
- Mac 앱이 불완전한 모델에서 안전하게 폴백하는가.

---

### Task 1: 학습 매니페스트 검증 규칙

**Files:**
- Create: `src/voice_tts/training/__init__.py`
- Create: `src/voice_tts/training/manifest.py`
- Create: `tests/test_training_manifest.py`

**Step 1: Write the failing tests**

- `ExcludedRange`, `CandidateSegment`, `ReviewRecord`를 요구하는 테스트를 추가한다.
- `[289.0, 311.0]`과 겹치는 `288.5–290.5`, `310.5–312.0`가 거절되는지 검증한다.
- 정규화 유사도 0.92 미만은 자동 승인하지 않으며, 승인 항목만 Qwen JSONL로 변환되는지 검증한다.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training_manifest.py -q`

Expected: 모듈/규칙 부재로 실패.

**Step 3: Implement the minimal production code**

- `GOLF_EXCLUSION = ExcludedRange(289.0, 311.0)`, `AUTO_APPROVE_SIMILARITY = 0.92`를 정의한다.
- `overlaps_exclusion`, 텍스트 정규화, `normalized_similarity`, `build_review_record`, `approved_records_to_jsonl`을 구현한다.
- JSONL은 `audio`, `text`, `ref_audio`만 가지며 ASR·거절 사유는 검토본에만 남긴다.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_training_manifest.py -q`

Expected: 매니페스트 규칙 테스트 통과.

**Step 5: Commit**

```bash
git add src/voice_tts/training tests/test_training_manifest.py
git commit -m 'feat: add voice training manifest validation'
```

### Task 2: Windows 데이터 준비와 검토 워크플로

**Files:**
- Create: `training_windows/prepare_dataset.py`
- Create: `training_windows/review_manifest.py`
- Create: `training_windows/requirements.txt`
- Create: `tests/test_training_windows_prepare.py`
- Modify: `.gitignore`

**Step 1: Write the failing tests**

- 48kHz M4A를 24kHz mono WAV로 변환할 작업과 VAD 후보가 2–12초로 묶이는지 테스트한다.
- 대본에 순차 매칭된 후보가 제외 범위와 겹치면 자동 거절되는지 검증한다.
- `approved: true`만 train/validation으로 승격되고, 매 5번째 적격 항목이 validation인지를 검증한다.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training_windows_prepare.py -q`

Expected: 준비 도구 부재로 실패.

**Step 3: Implement the minimal production code**

- `prepare_dataset.py`는 FFmpeg와 faster-whisper 단어 시간 정보를 사용해 후보를 만들고 `review_manifest.jsonl`에 시작/끝, 기대/ASR 대본, 유사도, 판정, 사유를 기록한다.
- `review_manifest.py`는 사람이 승인한 항목만 공통 참조 오디오를 포함한 Qwen train/validation JSONL로 쓴다. 빈 분할은 명확히 실패한다.
- CUDA PyTorch를 요구사항에 고정하지 않고, 오디오/ASR 패키지만 `requirements.txt`에 쓴다. `.gitignore`에 `training_data/`, `training_windows/output/`을 추가한다.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_training_windows_prepare.py tests/test_training_manifest.py -q`

Expected: 제외·검토·분할 규칙 통과.

**Step 5: Commit**

```bash
git add training_windows tests/test_training_windows_prepare.py .gitignore
git commit -m 'feat: add Windows voice dataset preparation workflow'
```

### Task 3: RTX 3070 Qwen 0.6B 학습 실행 도구

**Files:**
- Create: `training_windows/train_0_6b.py`
- Create: `training_windows/train.sh`
- Create: `training_windows/train.ps1`
- Create: `training_windows/README.md`
- Create: `tests/test_training_config.py`

**Step 1: Write the failing tests**

- `TrainingConfig.for_rtx_3070()`이 0.6B Base, batch 1, grad accumulation 8, 3 epoch, LR `2e-5`, speaker `dyson`, Korean인지 검증한다.
- 빈 데이터, CUDA/Flash Attention 사전 점검 실패는 학습 시작을 막고 active 모델을 변경하지 않는지 검증한다.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training_config.py -q`

Expected: 구성/사전점검 부재로 실패.

**Step 3: Implement the minimal production code**

- `train_0_6b.py`에 `nvidia-smi`, `torch.cuda.is_available()`, 8GB VRAM, flash-attn, train/validation JSONL preflight를 구현한다.
- Qwen 공식 finetuning을 호출하되 체크포인트는 `training_windows/output/run-<timestamp>/`에만 저장한다.
- 결과로 hyperparameter, 매니페스트 해시, checkpoint, 상태의 `training_report.json`을 쓴다.
- `train.sh`는 WSL2 진입점, `train.ps1`은 WSL 호출 진입점이다. README는 WSL2와 CUDA PyTorch 설치, OOM 시 중단 원칙을 설명한다.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_training_config.py -q`

Expected: 안전 구성과 실패 보호 테스트 통과.

**Step 5: Commit**

```bash
git add training_windows tests/test_training_config.py
git commit -m 'feat: add RTX 3070 Qwen training runner'
```

### Task 4: 고정 한국어 홀드아웃 평가와 내보내기 게이트

**Files:**
- Create: `training_windows/holdout_ko.txt`
- Create: `training_windows/evaluate.py`
- Create: `training_windows/activate_model.py`
- Create: `tests/test_training_export.py`

**Step 1: Write the failing tests**

- 고정된 5개 한국어 홀드아웃 문장이 학습 데이터와 분리되는지 테스트한다.
- 평가 보고서가 없거나 `passed`가 아니면 모델 내보내기/활성화를 차단하는지 검증한다.
- 성공 내보내기에 `model_manifest.json`과 `speaker: dyson`, `language: Korean`, 모델 식별자, 평가 보고서 해시가 있는지 검증한다.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_training_export.py -q`

Expected: 평가/활성화 게이트 부재로 실패.

**Step 3: Implement the minimal production code**

- `holdout_ko.txt`에 숫자·문장부호·일상 표현을 포함한 정확히 5개 문장을 작성한다.
- `evaluate.py`는 custom voice로 각 문장을 생성하고 WAV 및 `comparison_report.json`을 저장한다. 생성 실패, 누락 오디오, 잘못된 샘플레이트는 실패다.
- `activate_model.py`는 평가 통과 시에만 `data/models/fine_tuned/dyson/active/`에 새 모델을 내보내고, 완전성 확인 후 `model_manifest.json`을 기록한다.
- 사람의 청취 평가는 자동 합격으로 위장하지 않는다. 자동 검사 결과와 청취 확인 필요를 구분한다.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_training_export.py -q`

Expected: 평가 없는 활성화가 차단되고 유효 모델만 내보낼 수 있다.

**Step 5: Commit**

```bash
git add training_windows tests/test_training_export.py
git commit -m 'feat: add fine-tuned voice evaluation and export gate'
```

### Task 5: Mac 웹앱에 학습 모델 우선 엔진과 안전한 폴백 추가

**Files:**
- Modify: `src/voice_tts/config.py`
- Modify: `src/voice_tts/services/tts.py`
- Modify: `src/voice_tts/app.py`
- Modify: `tests/test_tts.py`
- Modify: `tests/test_api.py`

**Step 1: Write the failing tests**

- 유효한 `data/models/fine_tuned/dyson/active/model_manifest.json`이 있으면 `preferred_engine()`이 `FineTunedQwenTtsEngine`을 고르는지 테스트한다.
- 모델 폴더/매니페스트/speaker/language가 유효하지 않으면 zero-shot `QwenTtsEngine`으로 폴백하는지 검증한다.
- `GET /api/model`이 모드 `fine_tuned` 또는 `zero_shot`과 이유를 반환하는지, custom voice가 Korean/dyson으로 호출되는지 테스트한다.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tts.py tests/test_api.py -q`

Expected: 새 엔진 선택과 상태 API 부재로 실패.

**Step 3: Implement the minimal production code**

- `ProjectPaths`에 `fine_tuned_model_dir`을 추가한다.
- `FineTunedQwenTtsEngine`이 활성 모델을 명시적으로 로드하고 `generate_custom_voice(text=..., language="Korean", speaker="dyson")`를 호출하게 한다.
- 유효성 검사와 `preferred_engine(paths)` 팩토리를 추가한다. 기존 `QwenTtsEngine` zero-shot 경로는 바꾸지 않는다.
- 앱 초기화가 팩토리를 사용하게 하고 `/api/model` 상태 API를 추가한다.
- 모델을 로드하지 못해도 서버를 중단하지 않고 zero-shot에 머물며 상태 API에 폴백 사유를 보인다.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tts.py tests/test_api.py -q`

Expected: 학습 모델 선택, 폴백, API 상태 테스트 통과.

**Step 5: Commit**

```bash
git add src/voice_tts/config.py src/voice_tts/services/tts.py src/voice_tts/app.py tests/test_tts.py tests/test_api.py
git commit -m 'feat: load fine-tuned Korean voice model when available'
```

### Task 6: UI 상태와 로컬 전달 문서

**Files:**
- Modify: `src/voice_tts/static/index.html`
- Modify: `src/voice_tts/static/app.js`
- Modify: `src/voice_tts/static/style.css`
- Modify: `README.md`
- Modify: `tests/test_api.py`

**Step 1: Write the failing tests**

- 모델 상태 API가 UI에 필요한 mode와 fallback reason을 직렬화하는지 테스트한다.
- README에 비클라우드 정책, Windows active 모델 폴더의 Mac 복사 절차, 모델 제거 시 zero-shot 복귀 절차가 있는지 점검 기준을 추가한다.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py -q`

Expected: UI 상태 데이터가 부족해 실패.

**Step 3: Implement the minimal production code**

- 페이지 시작 시 `/api/model`을 조회해 `학습된 내 목소리` 또는 `기본 음성 복제`와 짧은 사유를 표시한다.
- 기존 생성과 다운로드 동작은 그대로 둔다.
- README에 Windows WSL2 준비, 데이터 검토, 학습, 평가, 모델 전달, Mac 재시작과 폴백을 순서대로 문서화한다.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py -q`

Expected: 상태 API 계약 통과 및 UI/문서 점검 기준 충족.

**Step 5: Commit**

```bash
git add src/voice_tts/static README.md tests/test_api.py
git commit -m 'feat: show active voice model status in local app'
```

### Task 7: 실제 Windows 학습 실행 및 수락 절차

**Files:**
- Create at run time: `training_windows/output/run-<timestamp>/training_report.json`
- Create at run time: `training_windows/output/run-<timestamp>/comparison_report.json`
- Create at run time: `data/models/fine_tuned/dyson/active/model_manifest.json`

**Step 1: Validate Windows preconditions**

- Windows PC에서 WSL2 Ubuntu, `nvidia-smi`, CUDA PyTorch, FFmpeg, Qwen 공식 finetuning 의존성을 확인한다.
- 새 M4A와 `recording_script_10min_ko.txt`를 Windows 작업 폴더에 복사한다.

**Step 2: Prepare and review data**

Run: `python prepare_dataset.py --audio <new-recording.m4a> --script <recording_script_10min_ko.txt>`

- `review_manifest.jsonl`에서 289–311초와 낮은 유사도 후보를 승인하지 않는다.
- 승인한 항목만 `review_manifest.py`로 train/validation JSONL로 승격한다.

**Step 3: Run preflight and train**

Run: `powershell -ExecutionPolicy Bypass -File .\train.ps1`

Expected: preflight 통과 뒤 run 폴더에 checkpoint와 training report가 생성된다. OOM/패키지 호환성 실패면 멈추고 active 모델을 만들지 않는다.

**Step 4: Evaluate, listen, and export**

- `evaluate.py`로 5개 홀드아웃 WAV를 생성한다.
- 사용자가 문장 일치와 음색 유사도를 청취 확인한다.
- 만족할 때만 `activate_model.py`로 active 폴더를 만들고 Mac의 `data/models/fine_tuned/dyson/active/`로 복사한다.

**Step 5: Verify Mac integration**

Run: `.venv/bin/pytest -q`

Expected: 전체 테스트 통과. `/api/model`은 `fine_tuned`를 반환하고, 결과물은 사용자의 청취 수락을 받아야 한다.

**Step 6: Commit**

- 실행 로그와 음성/모델 바이너리는 커밋하지 않는다. 코드/문서 수정 때만 별도 커밋한다.

## Final Verification

```bash
cd /Users/dyson/VSCode_src/my_voice_tts
.venv/bin/pytest -q
git diff --check
git status --short
```

- 실제 학습 성공은 `training_report.json`과 사용자의 청취 평가로만 주장한다.
- 모델이 없거나 검증에 실패한 환경에서도 기존 Korean zero-shot TTS가 정상 동작해야 한다.
