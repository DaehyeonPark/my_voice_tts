"""Serialized local synthesis jobs with atomic output creation."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable
from uuid import uuid4

from voice_tts.config import ProjectPaths
from voice_tts.services.audio import concatenate_wavs
from voice_tts.services.reference import Reference
from voice_tts.services.text import validate_and_chunk_text
from voice_tts.services.tts import SynthesisError, TtsEngine


@dataclass(frozen=True)
class GeneratedAudio:
    output_id: str
    path: Path


class SynthesisService:
    def __init__(
        self,
        paths: ProjectPaths,
        reference: Reference,
        engine: TtsEngine,
        max_chunk_chars: int,
        concatenate: Callable[[list[Path], Path], None] = concatenate_wavs,
    ) -> None:
        self._paths = paths
        self._reference = reference
        self._engine = engine
        self._max_chunk_chars = max_chunk_chars
        self._concatenate = concatenate
        self._lock = Lock()

    def generate(self, text: str) -> GeneratedAudio:
        with self._lock:
            return self._generate(text)

    def _generate(self, text: str) -> GeneratedAudio:
        chunks = validate_and_chunk_text(text, self._max_chunk_chars)
        self._paths.ensure_runtime_directories()
        output_id = uuid4().hex
        job_dir = self._paths.temp_dir / output_id
        job_dir.mkdir()
        final_path = self._paths.output_dir / f"{output_id}.wav"
        try:
            chunk_paths: list[Path] = []
            for index, chunk in enumerate(chunks, start=1):
                chunk_path = job_dir / f"{index:04d}.wav"
                try:
                    self._engine.synthesize(self._reference.path, self._reference.transcript, chunk, chunk_path)
                except Exception as error:
                    raise SynthesisError(f"{index}번째 문장을 생성하지 못했습니다.") from error
                chunk_paths.append(chunk_path)
            joined_path = job_dir / "joined.wav"
            self._concatenate(chunk_paths, joined_path)
            if not joined_path.is_file() or not joined_path.stat().st_size:
                raise SynthesisError("생성된 WAV 파일이 비어 있습니다.")
            joined_path.replace(final_path)
            return GeneratedAudio(output_id=output_id, path=final_path)
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
