# Windows / WSL2 Qwen3-TTS 학습

이 폴더는 사용자의 음성·모델을 외부로 전송하지 않는다. Windows PC의 WSL2 Ubuntu에서만 실행하며, 결과 체크포인트와 WAV는 Git에 커밋하지 않는다.

## 1. WSL2 환경 준비

Ubuntu에서 NVIDIA CUDA가 보이는지 먼저 확인한다.

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

프로젝트 루트에서 Python 가상환경을 만들고 앱·전처리 패키지를 설치한다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
pip install -r training_windows/requirements.txt
```

Qwen 공식 학습 코드를 로컬에 복제한다. `flash-attn`은 CUDA, PyTorch, 컴파일러 버전에 맞게 설치해야 한다.

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git ~/src/Qwen3-TTS
pip install qwen-tts accelerate safetensors tensorboard flash-attn --no-build-isolation
```

## 2. 데이터 준비과 검토

새 M4A와 `recording_script_10min_ko.txt`를 WSL 작업 폴더에 복사한다. 복사본은 `training_data/`에 두며 Git이 무시한다.

```bash
source .venv/bin/activate
python -m training_windows.prepare_dataset \
  --audio training_data/new-recording.m4a \
  --script recording_script_10min_ko.txt \
  --output-dir training_data
```

`training_data/review_manifest.jsonl`을 열어 사람이 `approved`만 `true`로 바꾼다. 289.0–311.0초의 골프 발화와 낮은 유사도 항목은 승인하지 않는다. 깨끗한 10–20초 WAV를 `training_data/reference.wav`로 만든 뒤 승인 항목을 승격한다.

```bash
python -m training_windows.review_manifest \
  --manifest training_data/review_manifest.jsonl \
  --reference-audio training_data/reference.wav \
  --output-dir training_data
```

## 3. 학습

RTX 3070 8GB 설정은 Qwen 0.6B, batch 1, gradient accumulation 8, 3 epoch, learning rate `2e-5`다. 공식 Qwen SFT가 누적 단계를 4로 고정한 버전이면 실행기는 원본을 `sft_12hz.py.before-dyson-patch`으로 한 번 백업하고 8로만 바꾼다.

```bash
export QWEN_FINETUNING_DIR="$HOME/src/Qwen3-TTS/finetuning"
bash training_windows/train.sh
```

Windows PowerShell에서는 WSL 경로를 명시해 실행한다.

```powershell
.\training_windows\train.ps1 -WslProjectPath "/home/<user>/src/my_voice_tts" -QwenFinetuningDir "/home/<user>/src/Qwen3-TTS/finetuning"
```

실행 전 CUDA, 8GB VRAM, flash-attn, 비어 있지 않은 train/validation JSONL을 검사한다. 하나라도 실패하거나 OOM이면 `training_windows/output/run-*/training_report.json`에 실패를 남기고 `data/models/fine_tuned/dyson/active/`는 변경하지 않는다.

학습 성공 뒤에도 active 모델은 자동으로 만들지 않는다. `evaluate.py`와 `activate_model.py`로 평가·청취한 모델만 활성화한다.
