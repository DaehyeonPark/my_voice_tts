from pathlib import Path

import pytest

from voice_tts.config import ProjectPaths
from voice_tts.services.reference import ReferenceStore


class WritingAudio:
    def prepare_reference(self, source: Path, start: float, duration: float, destination: Path) -> None:
        destination.write_bytes(b"wav")


def test_reference_store_rejects_blank_transcript(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    paths.study_source_dir.mkdir()
    (paths.study_source_dir / "voice.m4a").write_bytes(b"source")

    with pytest.raises(ValueError, match="대본"):
        ReferenceStore(paths, WritingAudio()).prepare("voice.m4a", 0, 15, " ")


def test_reference_store_keeps_source_and_saves_metadata(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    paths.study_source_dir.mkdir()
    source = paths.study_source_dir / "voice.m4a"
    source.write_bytes(b"source")

    reference = ReferenceStore(paths, WritingAudio()).prepare("voice.m4a", 0, 15, "안녕하세요.")

    assert source.read_bytes() == b"source"
    assert reference.path.read_bytes() == b"wav"
    assert reference.transcript == "안녕하세요."
