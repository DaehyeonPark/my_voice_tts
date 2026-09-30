"""Persistent metadata for the one local reference voice."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from voice_tts.config import ProjectPaths
from voice_tts.services.audio import prepare_reference


@dataclass(frozen=True)
class Reference:
    path: Path
    transcript: str


class ReferencePreparer(Protocol):
    def prepare_reference(self, source: Path, start: float, duration: float, destination: Path) -> None: ...


class AudioReferencePreparer:
    def prepare_reference(self, source: Path, start: float, duration: float, destination: Path) -> None:
        prepare_reference(source, start, duration, destination)


class ReferenceStore:
    def __init__(self, paths: ProjectPaths, preparer: ReferencePreparer | None = None) -> None:
        self._paths = paths
        self._preparer = preparer or AudioReferencePreparer()

    def prepare(self, source_name: str, start_seconds: float, duration_seconds: float, transcript: str) -> Reference:
        cleaned_transcript = " ".join(transcript.split())
        if not cleaned_transcript:
            raise ValueError("참조 음성의 정확한 대본을 입력해 주세요.")
        source = self._source_path(source_name)
        if not source.is_file():
            raise FileNotFoundError(source_name)
        self._paths.ensure_runtime_directories()
        destination = self._paths.reference_dir / "reference.wav"
        self._preparer.prepare_reference(source, start_seconds, duration_seconds, destination)
        reference = Reference(path=destination, transcript=cleaned_transcript)
        (self._paths.reference_dir / "reference.json").write_text(
            json.dumps({"path": destination.name, "transcript": reference.transcript}, ensure_ascii=False),
            encoding="utf-8",
        )
        return reference

    def load(self) -> Reference:
        metadata_path = self._paths.reference_dir / "reference.json"
        if not metadata_path.is_file():
            raise FileNotFoundError("준비된 참조 음성이 없습니다.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        path = self._paths.reference_dir / metadata["path"]
        if not path.is_file():
            raise FileNotFoundError("준비된 참조 음성 파일이 없습니다.")
        return Reference(path=path, transcript=metadata["transcript"])

    def _source_path(self, source_name: str) -> Path:
        root = self._paths.study_source_dir.resolve()
        candidate = (root / source_name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise FileNotFoundError(source_name) from error
        return candidate
