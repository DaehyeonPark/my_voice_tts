"""Generate one local smoke-test WAV from an already prepared reference."""

from pathlib import Path

from voice_tts.config import ProjectPaths, Settings
from voice_tts.services.jobs import SynthesisService
from voice_tts.services.reference import ReferenceStore
from voice_tts.services.tts import QwenTtsEngine


def main() -> None:
    paths = ProjectPaths.from_root(Path(__file__).resolve().parents[1])
    reference = ReferenceStore(paths).load()
    result = SynthesisService(paths, reference, QwenTtsEngine(paths.model_dir), Settings().max_chunk_chars).generate(
        "안녕하세요 저는 아시안게임 배구를 보고 있습니다. 우리나라가 상당히 잘 하네요."
    )
    if not result.path.is_file() or result.path.stat().st_size == 0:
        raise RuntimeError("생성된 WAV 파일이 비어 있습니다.")
    print(result.path)


if __name__ == "__main__":
    main()
