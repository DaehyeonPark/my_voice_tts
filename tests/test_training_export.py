import json
from hashlib import sha256
from pathlib import Path

import pytest

from training_windows.activate_model import ActivationBlocked, activate_model
from training_windows.evaluate import load_holdouts


def test_holdout_contains_exactly_five_fixed_korean_sentences() -> None:
    holdouts = load_holdouts(Path("training_windows/holdout_ko.txt"))

    assert len(holdouts) == 5
    assert all(sentence.strip() for sentence in holdouts)
    assert any("2026" in sentence for sentence in holdouts)


def test_activation_requires_a_passed_and_listening_approved_evaluation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    active = tmp_path / "active"

    with pytest.raises(ActivationBlocked):
        activate_model(checkpoint, tmp_path / "missing-report.json", active)

    report = tmp_path / "comparison_report.json"
    report.write_text(json.dumps({"status": "review_required", "listening_approved": False, "automatic_checks_passed": True}), encoding="utf-8")
    with pytest.raises(ActivationBlocked):
        activate_model(checkpoint, report, active)


def test_successful_activation_writes_manifest_with_evaluation_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    report = tmp_path / "comparison_report.json"
    report.write_text(json.dumps({"status": "passed", "listening_approved": True, "automatic_checks_passed": True, "model_sha256": sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()}), encoding="utf-8")

    manifest_path = activate_model(checkpoint, report, tmp_path / "active")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["speaker"] == "dyson"
    assert manifest["language"] == "Korean"
    assert manifest["model_id"] == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    assert len(manifest["evaluation_report_sha256"]) == 64
