#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${QWEN_FINETUNING_DIR:?Set QWEN_FINETUNING_DIR to the cloned Qwen3-TTS/finetuning directory.}"

python -m training_windows.train_0_6b \
  --train-jsonl "${TRAIN_JSONL:-training_data/train.jsonl}" \
  --validation-jsonl "${VALIDATION_JSONL:-training_data/validation.jsonl}" \
  --output-root "${OUTPUT_ROOT:-training_windows/output}" \
  --finetuning-dir "${QWEN_FINETUNING_DIR}" \
  --active-model-dir "${ACTIVE_MODEL_DIR:-data/models/fine_tuned/dyson/active}"
