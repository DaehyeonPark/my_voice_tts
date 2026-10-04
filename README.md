# 내 목소리 TTS

내 Mac에서만 실행되는 한국어 음성 복제 TTS 웹앱입니다. 음성·입력 텍스트·생성 결과는 외부 TTS 서비스로 전송하지 않습니다. 첫 합성 때 한국어를 지원하는 Qwen3-TTS 음성 복제 모델 파일만 내려받습니다.

## 실행

**Windows에서 학습한 목소리를 MacBook Pro M4·16GB에서 사용하려면
[Mac 설치·모델 가져오기·실행 안내](mac/README.md)를 먼저 따르세요.**

```bash
bash mac/setup.sh
.venv/bin/python mac/import_voice.py ~/Downloads/my-voice-mac.tar
bash mac/run.sh
```

개인 모델은 Git에 포함되지 않으며 `my-voice-mac.tar`를 한 번 로컬 복사해야 합니다.
아래는 학습 모델 없이 기본 음성 복제를 처음 준비하는 절차입니다.

1. macOS에서 FFmpeg를 설치합니다: `brew install ffmpeg`
2. 프로젝트 폴더에서 가상환경과 의존성을 준비합니다.

   `python3.12 -m venv .venv`

   `.venv/bin/pip install -e '.[dev]'`

3. 제공한 녹음 파일을 `study_src/`에 둡니다. 이 폴더의 원본은 앱이 수정하지 않습니다.
4. `.venv/bin/voice-tts`를 실행하고 브라우저에서 `http://127.0.0.1:8765`를 엽니다.
5. 화면에서 깨끗한 10~20초 구간과 그 구간의 **정확한 대본**을 넣어 참조 음성을 준비한 뒤, 읽을 텍스트를 생성합니다.

WAV 결과는 `data/outputs/`에 저장됩니다. 앱은 모든 생성 요청에 한국어 언어 설정을 명시합니다. 참조 음성이나 결과가 어색하면 더 조용한 환경에서 10~15분 분량의 녹음을 추가해 주세요.

## 문제 해결

- `FFmpeg을 찾을 수 없습니다`: `brew install ffmpeg` 후 다시 실행하세요.
- 모델 준비 실패: 인터넷 연결을 확인한 뒤 다시 생성하세요. 모델이 내려받힌 뒤에는 오프라인으로 작동합니다.
- 준비된 참조 음성이 없다는 메시지: 화면에서 참조 음성을 먼저 준비하세요.

선택적 실제 합성 점검은 `.venv/bin/python scripts/smoke.py`로 실행합니다.

## Windows GPU로 음색 개선하기

`training_windows/README.md`의 WSL2 절차로 새 녹음과 대본을 검토·학습합니다. 이 과정에서도 오디오와 모델은 Mac/Windows PC 안에서만 이동하며 클라우드에 업로드하지 않습니다.

청취 평가를 통과한 Windows 결과의 `active/` 폴더 전체를 Mac의 `data/models/fine_tuned/dyson/active/`로 로컬 복사한 뒤 서버를 재시작하세요. 화면에 **학습된 내 목소리 사용 중**이 표시되면 적용된 것입니다.

되돌리려면 `data/models/fine_tuned/dyson/active/` 폴더를 다른 이름으로 옮기거나 삭제하고 서버를 재시작하세요. 앱은 자동으로 **기본 음성 복제 사용 중**으로 안전하게 돌아갑니다.
