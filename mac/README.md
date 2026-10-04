# MacBook Pro M4 · 메모리 16GB

Windows에서 청취 승인한 0.6B 모델을 그대로 사용한다. 재학습은 필요 없다.
코드와 설치 스크립트는 Git으로 받고, 개인 모델은 로컬 파일로 한 번 옮긴다.
**Git clone만으로 개인 목소리 가중치가 설치되지는 않는다.**

## 1. 코드와 실행 환경 설치

macOS 기본 터미널에서 실행한다. Homebrew가 없다면 https://brew.sh 의 안내로 먼저 설치한다.

```bash
git clone https://github.com/DaehyeonPark/my_voice_tts.git
cd my_voice_tts
bash mac/setup.sh
```

이미 clone했다면 해당 폴더에서 `git pull --ff-only` 후 설치한다.
설치기는 Python 3.12 arm64, FFmpeg, SoX와 프로젝트 전용 `.venv`를 준비한다.
NVIDIA CUDA / flash-attn은 설치하지 않는다.

## 2. 개인 모델 한 번 가져오기

Windows에서 준비된 `my-voice-mac.tar`를 USB/외장 SSD/로컬 공유로 Mac에 복사한다.
예를 들어 다운로드 폴더에 두었다면:

```bash
.venv/bin/python mac/import_voice.py ~/Downloads/my-voice-mac.tar
```

파일별 SHA256을 확인한 뒤 `data/models/fine_tuned/dyson/active/`와
`data/reference/`에 설치한다. 기존 모델이나 참조 폴더가 있으면 덮어쓰지 않고 중단한다.
기존 폴더를 별도 위치에 백업한 뒤 다시 가져온다. 압축 파일과 풀린 모델의 공간이 모두 필요하다.
이 파일에는 개인 음성 모델과 참조 녹음이 있으므로 Git에 추가하지 않는다.

Windows에서 다시 내보내야 할 때는 WSL 프로젝트 루트에서:

```bash
source training_windows/activate_env.sh
python mac/export_voice.py
```

결과는 `data/exports/my-voice-mac.tar`이다.

## 3. 실행과 영상 편집

```bash
bash mac/run.sh
```

브라우저에서 http://localhost:8765 를 연다. 텍스트 입력 → 생성 → WAV 다운로드.
참조 음성은 미리 가져왔으므로 새로 설정할 필요가 없다.
다운로드한 WAV를 Final Cut Pro, Premiere Pro, DaVinci Resolve 등의 오디오 트랙에 넣는다.
원본 출력은 프로젝트의 `data/outputs/`에도 남는다. 앱 종료는 터미널에서 Ctrl+C.

## M4 실행 설정과 한계

- Apple GPU의 MPS, float32, SDPA를 명시적으로 선택한다. 0.6B 체크포인트를 사용한다.
- 지원되지 않는 일부 MPS 연산에는 PyTorch의 CPU fallback을 허용한다.
- Mac 하드웨어에서 실제 합성/속도/최대 메모리는 아직 측정하지 않았다.
  첫 실행에서는 짧은 문장으로 확인하고, 16GB 메모리 부담을 줄이려면 무거운 편집 작업과
  대량 합성을 동시에 하지 않는다. 앱은 문단을 나누어 처리하지만 긴 입력은 시간이 걸린다.
- MPS 오류가 발생하면 서버를 Ctrl+C로 종료한 뒤 아래 명령으로 CPU에서 실행할 수 있다.
  CPU는 느릴 수 있으며, 이 옵션은 사용자가 명시적으로 선택한다.

```bash
VOICE_TTS_DEVICE=cpu bash mac/run.sh
```

- 모델 파일이 정상 설치되면 학습된 모델은 로컬에서 읽는다. 로딩 실패 시 기존 앱의
  기본 음성 복제 fallback이 동작할 수 있으므로 `/api/model`에서 `fine_tuned`인지 확인한다.
- 장비별 출력 차이가 있을 수 있으므로 Mac에서 처음 생성한 음성은 편집 전에 들어본다.

참고: [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html),
[Qwen3-TTS 공식 로딩 API](https://github.com/QwenLM/Qwen3-TTS).
