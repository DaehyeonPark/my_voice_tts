"""Promote only human-approved review entries into Qwen training JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from voice_tts.training.manifest import CandidateSegment, GOLF_EXCLUSION, overlaps_exclusion


def _eligible(entry: dict[str, object]) -> bool:
    if entry.get("approved") is not True or entry.get("reason") is not None:
        return False
    segment = CandidateSegment(
        float(entry["start_seconds"]),
        float(entry["end_seconds"]),
        str(entry["expected_text"]),
        str(entry.get("observed_text", "")),
    )
    return not overlaps_exclusion(segment, GOLF_EXCLUSION)


def promote_approved_entries(
    entries: Iterable[dict[str, object]], *, reference_audio: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    approved = [entry for entry in entries if _eligible(entry)]
    train_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    for index, entry in enumerate(approved, start=1):
        row = {
            "audio": str(entry["audio"]),
            "text": str(entry["expected_text"]),
            "ref_audio": reference_audio,
        }
        (validation_rows if index % 5 == 0 else train_rows).append(row)
    if not train_rows or not validation_rows:
        raise ValueError("승인된 발화가 부족합니다. train과 validation에 각각 하나 이상 필요합니다.")
    return train_rows, validation_rows


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: Iterable[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def promote_manifest(manifest_path: Path, reference_audio: Path, output_dir: Path) -> tuple[Path, Path]:
    train_rows, validation_rows = promote_approved_entries(read_jsonl(manifest_path), reference_audio=str(reference_audio))
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    write_jsonl(train_rows, train_path)
    write_jsonl(validation_rows, validation_path)
    return train_path, validation_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote reviewed speech candidates into Qwen JSONL datasets.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("training_data"))
    args = parser.parse_args()
    train_path, validation_path = promote_manifest(args.manifest, args.reference_audio, args.output_dir)
    print(f"train={train_path}\nvalidation={validation_path}")


if __name__ == "__main__":
    main()
