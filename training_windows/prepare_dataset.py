"""Prepare reviewable speech candidates from a local recording in WSL2."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import subprocess
from typing import Iterable

from voice_tts.training.manifest import CandidateSegment, build_review_record, normalized_similarity


MIN_SEGMENT_SECONDS = 2.0
MAX_SEGMENT_SECONDS = 12.0
MAX_WORD_GAP_SECONDS = 0.8


@dataclass(frozen=True)
class WordTiming:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True)
class PreparedSegment:
    start_seconds: float
    end_seconds: float
    observed_text: str


def build_ffmpeg_command(source: str | Path, destination: str | Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ar",
        "24000",
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(destination),
    ]


def group_word_timings(
    words: Iterable[WordTiming],
    *,
    min_seconds: float = MIN_SEGMENT_SECONDS,
    max_seconds: float = MAX_SEGMENT_SECONDS,
    max_gap_seconds: float = MAX_WORD_GAP_SECONDS,
) -> list[PreparedSegment]:
    """Group contiguous ASR words into reviewable 2–12 second candidates."""

    groups: list[PreparedSegment] = []
    current: list[WordTiming] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        duration = current[-1].end_seconds - current[0].start_seconds
        if min_seconds <= duration <= max_seconds:
            groups.append(PreparedSegment(current[0].start_seconds, current[-1].end_seconds, "".join(word.text for word in current).strip()))
        current = []

    for word in words:
        if word.end_seconds <= word.start_seconds or not word.text.strip():
            continue
        if current:
            exceeds_duration = word.end_seconds - current[0].start_seconds > max_seconds
            has_gap = word.start_seconds - current[-1].end_seconds > max_gap_seconds
            if exceeds_duration or has_gap:
                flush()
        current.append(word)
    flush()
    return groups


def split_script_sentences(script: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", script) if sentence.strip()]


def assign_expected_texts(segments: list[PreparedSegment], script: str) -> list[str]:
    sentences = split_script_sentences(script)
    if not sentences:
        raise ValueError("대본에 읽을 문장이 없습니다.")
    expected: list[str] = []
    cursor = 0
    for segment in segments:
        candidates = [" ".join(sentences[cursor : cursor + width]) for width in range(1, min(3, len(sentences) - cursor) + 1)]
        if not candidates:
            expected.append("")
            continue
        best = max(candidates, key=lambda text: normalized_similarity(text, segment.observed_text))
        expected.append(best)
        cursor += len(split_script_sentences(best))
    return expected


def build_review_entry(segment: PreparedSegment, *, expected_text: str, audio_path: str) -> dict[str, object]:
    record = build_review_record(
        CandidateSegment(segment.start_seconds, segment.end_seconds, expected_text, segment.observed_text)
    )
    return {
        "audio": audio_path,
        "start_seconds": segment.start_seconds,
        "end_seconds": segment.end_seconds,
        "expected_text": expected_text,
        "observed_text": segment.observed_text,
        "similarity": record.similarity,
        "auto_approved": record.auto_approved,
        "approved": record.approved,
        "reason": record.reason,
    }


def transcode_to_wav(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(build_ffmpeg_command(source, destination), check=True)


def transcribe_words(wav_path: Path, model_name: str) -> list[WordTiming]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    segments, _ = model.transcribe(str(wav_path), language="ko", word_timestamps=True)
    words: list[WordTiming] = []
    for segment in segments:
        for word in segment.words or []:
            if word.start is not None and word.end is not None:
                words.append(WordTiming(float(word.start), float(word.end), word.word))
    return words


def write_review_manifest(entries: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries), encoding="utf-8")


def prepare_dataset(audio: Path, script_path: Path, output_dir: Path, whisper_model: str) -> Path:
    wav_path = output_dir / "processed" / "recording_24khz_mono.wav"
    transcode_to_wav(audio, wav_path)
    prepared = group_word_timings(transcribe_words(wav_path, whisper_model))
    expected_texts = assign_expected_texts(prepared, script_path.read_text(encoding="utf-8"))
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for index, (segment, expected_text) in enumerate(zip(prepared, expected_texts), start=1):
        clip_path = clips_dir / f"{index:04d}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(segment.start_seconds), "-to", str(segment.end_seconds), "-i", str(wav_path), str(clip_path)],
            check=True,
        )
        entries.append(build_review_entry(segment, expected_text=expected_text, audio_path=str(clip_path)))
    manifest_path = output_dir / "review_manifest.jsonl"
    write_review_manifest(entries, manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a review manifest for local Korean voice fine-tuning.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("training_data"))
    parser.add_argument("--whisper-model", default="medium")
    args = parser.parse_args()
    print(prepare_dataset(args.audio, args.script, args.output_dir, args.whisper_model))


if __name__ == "__main__":
    main()
