# Windows GPU 기반 개인 한국어 음성 파인튜닝 설계

## 목표

사용자의 새 한국어 녹음으로 단일 화자 Qwen3-TTS 모델을 미세조정해 다음을 개선한다.

1. 입력한 한국어 문장과 생성 음성의 내용 일치도를 높인다.
2. 생성 음색을 사용자의 실제 음색에 더 가깝게 만든다.

생성 웹앱은 계속 Mac에서 `127.0.0.1`으로만 실행한다. 학습은 사용자의
Windows NVIDIA GPU PC에서만 수행하며, 음성이나 모델을 제3자 클라우드에
업로드하지 않는다.

## 확정된 입력과 제약

- 원본 녹음: `/Users/dyson/Downloads/대현TTS_두번째녹음.m4a`
- 원본 형식: mono AAC, 48 kHz, 374.1초
- 원본 대본: `recording_script_10min_ko.txt`
- 대본 외 골프 발화: 04:50부터 05:10까지. 이 20초는 모든 학습·검증 데이터에서 제외한다.
- Windows 학습 GPU: NVIDIA GeForce RTX 3070, VRAM 8 GB, CUDA 12.6 드라이버.
- 이 메모리에서는 1.7B 전체 SFT를 시도하지 않는다. 공식 지원 모델인
  `Qwen/Qwen3-TTS-12Hz-0.6B-Base`를 batch size 1로 학습한다.

## 선택한 접근

Qwen3-TTS 공식 단일 화자 SFT 흐름을 따른다. 공식 포맷은 음성 경로, 정확한
대본, 동일한 참조 음성을 가진 JSONL이며, 학습 결과는 `CustomVoice` 모델로
사용한다. 따라서 기존의 즉석 음성 복제 호출과 학습 모델의 CustomVoice 호출을
명시적으로 분리한다.

공식 학습 코드가 CUDA와 FlashAttention 2를 전제로 하므로 Windows에서는
WSL2 Ubuntu 환경에서 실행한다. RTX 3070의 메모리 한계를 고려해 batch size 1,
gradient accumulation 8, 짧은 음성 샘플, 낮은 학습률, 초기 3 epoch를 사용한다.
첫 trial이 OOM이면 임의의 대체 학습을 시도하지 않고 진단 결과를 남기고 최소
12 GB VRAM GPU가 필요하다고 보고한다.

## 데이터 흐름

```text
Mac 원본 m4a + 확정 대본
  -> 학습 패키지로 복사 (Git 제외)
  -> Windows WSL2의 전처리
       -> 04:50~05:10 제거
       -> 16-bit mono WAV로 변환
       -> VAD/ASR 기반 2~12초 발화 후보 분할
       -> 대본 순차 정렬 및 불일치 후보 격리
       -> 사람이 확인할 review manifest
       -> 승인된 train/validation JSONL
  -> Qwen tokenizer audio_codes 추출
  -> Qwen 0.6B 단일 화자 SFT
  -> epoch별 checkpoint와 검증 결과
  -> 선택 checkpoint를 Mac data/models/fine_tuned/dyson/로 복사
  -> Mac 웹앱이 학습 CustomVoice 모델로 생성
```

### 데이터 품질 규칙

- 골프 발화 시간과 그 경계 1초는 제외한다.
- `recording_script_10min_ko.txt`와 정규화 후 일치하지 않는 ASR 구간은 자동으로
  학습 데이터에 넣지 않는다.
- review manifest에는 시간, 원문 대본, ASR 대본, 일치 점수, 제외 사유를 기록한다.
- 사용자가 승인한 샘플만 `train_raw.jsonl`로 승격한다.
- 동일한 깨끗한 10~20초 WAV를 모든 JSONL 레코드의 `ref_audio`로 사용한다.
- 검증용 문장은 학습에서 제외해, 단순 암기가 아닌 생성 품질을 확인한다.

## Windows 학습 패키지

프로젝트의 `training_windows/`에 다음을 둔다.

- `README.md`: WSL2, NVIDIA CUDA, Python 환경, 디스크 용량과 실행 절차
- `requirements.txt`: Qwen3-TTS 학습과 전처리에 필요한 고정 의존성
- `prepare_dataset.py`: 시간 제외, WAV 변환, 후보 분할, 대본 정렬, manifest 생성
- `review_manifest.py`: 승인된 샘플만 train/validation JSONL로 만드는 비대화형 도구
- `train.ps1`: WSL2에서 학습을 시작하는 Windows 진입점
- `train.sh`: 0.6B checkpoint, batch size 1, accumulation 8의 Qwen 학습 실행
- `evaluate.py`: 고정된 한국어 holdout 문장을 생성하고 비교용 WAV와 보고서를 작성

원본 오디오, 변환 WAV, JSONL, checkpoint와 출력 WAV는 `training_data/` 및
`data/`에만 두고 Git ignore 처리한다.

## Mac 웹앱 변경

- `FineTunedQwenTtsEngine`을 추가한다. 이 엔진은
  `generate_custom_voice(text=..., language="Korean", speaker="dyson")`를 호출한다.
- 학습 모델 설정 파일이 있으면 학습 엔진을 우선 사용하고, 없거나 로드에 실패하면
  현재의 zero-shot `QwenTtsEngine`을 사용한다.
- 앱 화면에는 현재 사용 중인 모델이 `학습된 내 목소리` 또는 `기본 음성 복제`인지
  표시한다.
- 학습 모델이 생성한 음성은 입력 텍스트와 무관한 참조 대본을 필요로 하지 않는다.

## 검증 기준

1. 전처리 테스트: 290~310초 샘플이 manifest와 JSONL 어느 곳에도 포함되지 않는다.
2. 대본 정렬 테스트: 불일치 샘플이 자동으로 보류되고 승인된 샘플만 학습 JSONL에
   포함된다.
3. 엔진 테스트: 학습 모델이 `Korean`과 `dyson` speaker를 사용해 호출된다.
4. 학습 전후 비교: 학습에 포함하지 않은 한국어 문장 5개를 두 모델로 생성한다.
   기준 음성과 새 결과를 함께 남겨 사용자가 발음 정확도와 음색을 청취 비교한다.
5. 회귀: 기존 기본 음성 복제 API·다운로드·로컬 전용 바인딩 테스트가 통과한다.

## 실패 처리와 안전성

- CUDA/FlashAttention 설치 실패, VRAM OOM, 대본 정렬 불량은 명확한 보고서로
  중단하며 부정확한 모델을 Mac 앱의 기본 모델로 승격하지 않는다.
- 모델 파일은 Windows와 Mac 사이에서 사용자가 선택한 로컬 전송 경로로만 이동한다.
- 파인튜닝에 성공한 checkpoint만 `data/models/fine_tuned/dyson/active`로 활성화한다.
- 활성화 이전 버전은 보존해 앱에서 즉시 기본 음성 복제로 되돌릴 수 있다.
