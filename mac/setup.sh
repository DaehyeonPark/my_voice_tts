#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo 'Apple Silicon Mac의 기본 터미널에서 실행하세요 (Rosetta 사용 안 함).'; exit 1
fi
if ! command -v brew >/dev/null; then
  echo 'https://brew.sh 에서 Homebrew를 설치한 뒤 다시 실행하세요.'; exit 1
fi
brew install python@3.12 ffmpeg sox
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r mac/requirements.txt -e .
echo '설치 완료. 모델을 가져온 뒤 bash mac/run.sh 를 실행하세요.'
