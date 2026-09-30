from pathlib import Path

import pytest

from voice_tts.services.audio import AudioPreparationError, concatenate_wavs, prepare_reference


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, command: list[str]) -> None:
        self.calls.append(command)


class FailingRunner:
    def run(self, command: list[str]) -> None:
        raise RuntimeError("conversion failed")


def test_prepare_reference_requests_24k_mono_wav(tmp_path: Path) -> None:
    runner = RecordingRunner()

    prepare_reference(Path("input.m4a"), 0, 15, tmp_path / "reference.wav", runner)

    assert runner.calls[0] == [
        "ffmpeg",
        "-y",
        "-ss",
        "0",
        "-i",
        "input.m4a",
        "-t",
        "15",
        "-ac",
        "1",
        "-ar",
        "24000",
        str(tmp_path / "reference.wav"),
    ]


def test_ffmpeg_failure_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(AudioPreparationError, match="FFmpeg"):
        prepare_reference(Path("input.m4a"), 0, 15, tmp_path / "reference.wav", FailingRunner())


def test_concatenate_wavs_uses_a_list_file(tmp_path: Path) -> None:
    runner = RecordingRunner()
    destination = tmp_path / "joined.wav"

    concatenate_wavs([Path("first.wav"), Path("second.wav")], destination, runner)

    assert runner.calls[0][0:4] == ["ffmpeg", "-y", "-f", "concat"]
    assert runner.calls[0][-1] == str(destination)
