#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/voice-tts ]]; then
  echo '먼저 bash mac/setup.sh 를 실행하세요.'; exit 1
fi
if [[ ! -f data/models/fine_tuned/dyson/active/model_manifest.json || ! -f data/reference/reference.json ]]; then
  echo '개인 모델을 먼저 가져오세요: .venv/bin/python mac/import_voice.py /경로/my-voice-mac.tar'; exit 1
fi
export VOICE_TTS_DEVICE="${VOICE_TTS_DEVICE:-mps}"
# Unsupported individual Metal operations may execute on CPU.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
echo 'http://localhost:8765 를 여세요. 종료: Ctrl+C'
exec .venv/bin/voice-tts
