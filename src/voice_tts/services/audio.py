"""FFmpeg-backed local audio preparation helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol


class AudioPreparationError(RuntimeError):
    """Raised when local audio tooling cannot prepare an audio file."""


class CommandRunner(Protocol):
    def run(self, command: list[str]) -> None: ...


class SubprocessRunner:
    def run(self, command: list[str]) -> None:
        subprocess.run(command, check=True, capture_output=True, text=True)


def prepare_reference(
    source: Path,
    start_seconds: float,
    duration_seconds: float,
    destination: Path,
    runner: CommandRunner | None = None,
) -> None:
    """Convert one clean source range to 24 kHz mono WAV."""
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        _format_seconds(start_seconds),
        "-i",
        str(source),
        "-t",
        _format_seconds(duration_seconds),
        "-ac",
        "1",
        "-ar",
        "24000",
        str(destination),
    ]
    _run(command, runner or SubprocessRunner())


def concatenate_wavs(
    inputs: list[Path],
    destination: Path,
    runner: CommandRunner | None = None,
) -> None:
    """Join WAV chunks in order with FFmpeg's concat demuxer."""
    if not inputs:
        raise AudioPreparationError("합칠 오디오가 없습니다.")
    list_file = destination.with_suffix(".concat.txt")
    list_file.write_text("".join(f"file '{path.resolve()}'\n" for path in inputs), encoding="utf-8")
    command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(destination)]
    try:
        _run(command, runner or SubprocessRunner())
    finally:
        list_file.unlink(missing_ok=True)


def _run(command: list[str], runner: CommandRunner) -> None:
    try:
        runner.run(command)
    except FileNotFoundError as error:
        raise AudioPreparationError("FFmpeg을 찾을 수 없습니다. Homebrew로 ffmpeg를 설치해 주세요.") from error
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        raise AudioPreparationError("FFmpeg이 오디오를 변환하지 못했습니다. 파일과 설치 상태를 확인해 주세요.") from error


def _format_seconds(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)
