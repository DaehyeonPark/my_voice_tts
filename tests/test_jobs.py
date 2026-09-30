from pathlib import Path

import pytest

from voice_tts.config import ProjectPaths
from voice_tts.services.jobs import SynthesisError, SynthesisService
from voice_tts.services.reference import Reference


class FakeEngine:
    def synthesize(self, reference_audio: Path, reference_text: str, text: str, output_path: Path) -> None:
        if "실패" in text:
            raise RuntimeError("model error")
        output_path.write_bytes(text.encode())


def _join(inputs: list[Path], destination: Path) -> None:
    destination.write_bytes(b"".join(path.read_bytes() for path in inputs))


def test_failed_later_chunk_removes_partial_audio(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure_runtime_directories()
    reference_path = paths.reference_dir / "reference.wav"
    reference_path.write_bytes(b"reference")
    service = SynthesisService(paths, Reference(reference_path, "참조 대본"), FakeEngine(), 12, _join)

    with pytest.raises(SynthesisError, match="2번째 문장"):
        service.generate("첫 문장입니다. 실패 문장입니다.")

    assert list(paths.output_dir.glob("*.wav")) == []
    assert not any(paths.temp_dir.iterdir())


def test_successful_chunks_become_one_output_wav(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure_runtime_directories()
    reference_path = paths.reference_dir / "reference.wav"
    reference_path.write_bytes(b"reference")
    service = SynthesisService(paths, Reference(reference_path, "참조 대본"), FakeEngine(), 12, _join)

    generated = service.generate("첫 문장입니다. 둘째 문장입니다.")

    assert generated.path.parent == paths.output_dir
    assert generated.path.read_bytes() == "첫 문장입니다.둘째 문장입니다.".encode()
