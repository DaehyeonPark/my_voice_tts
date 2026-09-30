"""Activate only a locally evaluated and listener-approved fine-tuned model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import uuid


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


class ActivationBlocked(RuntimeError):
    pass


def _approved_report(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ActivationBlocked("평가 보고서가 없습니다.")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("listening_approved") is not True or report.get("automatic_checks_passed") is not True:
        raise ActivationBlocked("청취 승인된 통과 평가 보고서가 필요합니다.")
    return report


def activate_model(checkpoint: Path, evaluation_report: Path, active_dir: Path) -> Path:
    _approved_report(evaluation_report)
    if not (checkpoint / "model.safetensors").is_file():
        raise ActivationBlocked("checkpoint에 model.safetensors가 없습니다.")
    if _approved_report(evaluation_report).get("model_sha256") != sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest():
        raise ActivationBlocked("평가 보고서가 이 checkpoint의 가중치와 일치하지 않습니다.")
    active_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = active_dir.parent / f".active-stage-{uuid.uuid4().hex}"
    shutil.copytree(checkpoint, stage)
    manifest = {
        "model_id": MODEL_ID,
        "speaker": "dyson",
        "language": "Korean",
        "evaluation_report_sha256": sha256(evaluation_report.read_bytes()).hexdigest(),
    }
    manifest_path = stage / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if active_dir.exists():
        backup = active_dir.parent / f"previous-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        active_dir.replace(backup)
    stage.replace(active_dir)
    return active_dir / "model_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Activate an evaluated local Qwen voice model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--active-dir", type=Path, default=Path("data/models/fine_tuned/dyson/active"))
    args = parser.parse_args()
    print(activate_model(args.checkpoint, args.evaluation_report, args.active_dir))


if __name__ == "__main__":
    main()
