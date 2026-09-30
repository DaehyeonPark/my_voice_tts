"""Run Qwen3-TTS 0.6B single-speaker training only after local safety checks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable


MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
MINIMUM_VRAM_BYTES = 8 * 1024**3


@dataclass(frozen=True)
class TrainingConfig:
    train_jsonl: Path
    validation_jsonl: Path
    output_root: Path
    finetuning_dir: Path
    model_name: str = MODEL_NAME
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    speaker_name: str = "dyson"
    language: str = "Korean"

    @classmethod
    def for_rtx_3070(
        cls, *, train_jsonl: Path, validation_jsonl: Path, output_root: Path, finetuning_dir: Path
    ) -> "TrainingConfig":
        return cls(train_jsonl, validation_jsonl, output_root, finetuning_dir)


@dataclass(frozen=True)
class PreflightResult:
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


class PreflightFailed(RuntimeError):
    pass


def _default_command(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def _default_torch():
    import torch

    return torch


def _nonempty_jsonl(path: Path) -> bool:
    return path.is_file() and any(line.strip() for line in path.read_text(encoding="utf-8").splitlines())


def check_preflight(
    config: TrainingConfig,
    *,
    run_command: Callable[[list[str]], str] = _default_command,
    torch_module=None,
    module_available: Callable[[str], bool] | None = None,
) -> PreflightResult:
    issues: list[str] = []
    if not _nonempty_jsonl(config.train_jsonl):
        issues.append("train JSONL이 없거나 비어 있습니다.")
    if not _nonempty_jsonl(config.validation_jsonl):
        issues.append("validation JSONL이 없거나 비어 있습니다.")
    if not (config.finetuning_dir / "prepare_data.py").is_file() or not (config.finetuning_dir / "sft_12hz.py").is_file():
        issues.append("Qwen3-TTS finetuning 디렉터리에 prepare_data.py 또는 sft_12hz.py가 없습니다.")
    try:
        reported_megabytes = int(re.search(r"\d+", run_command(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])).group())
        if reported_megabytes * 1024**2 < MINIMUM_VRAM_BYTES:
            issues.append("GPU VRAM이 8GB보다 작습니다.")
    except Exception:
        issues.append("nvidia-smi로 GPU VRAM을 확인할 수 없습니다.")
    torch_module = torch_module or _default_torch()
    if not torch_module.cuda.is_available():
        issues.append("CUDA PyTorch를 사용할 수 없습니다.")
    elif torch_module.cuda.get_device_properties(0).total_memory < MINIMUM_VRAM_BYTES:
        issues.append("PyTorch가 보고한 GPU VRAM이 8GB보다 작습니다.")
    module_available = module_available or (lambda name: importlib.util.find_spec(name) is not None)
    if not module_available("flash_attn"):
        issues.append("flash-attn이 설치되어 있지 않습니다.")
    return PreflightResult(tuple(issues))


def _manifest_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_report(run_dir: Path, *, status: str, config: TrainingConfig, issues: tuple[str, ...] = (), checkpoint_path: Path | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": status,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "manifest_hashes": {
            "train": _manifest_hash(config.train_jsonl) if config.train_jsonl.is_file() else None,
            "validation": _manifest_hash(config.validation_jsonl) if config.validation_jsonl.is_file() else None,
        },
        "issues": list(issues),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }
    path = run_dir / "training_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _run_directory(output_root: Path) -> Path:
    return output_root / f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def patch_gradient_accumulation(sft_script: Path, steps: int) -> Path | None:
    """Patch only the official hard-coded Accelerator accumulation value, with a backup."""

    source = sft_script.read_text(encoding="utf-8")
    match = re.search(r"gradient_accumulation_steps=\d+", source)
    if not match:
        raise RuntimeError("Qwen SFT 스크립트에서 gradient_accumulation_steps를 찾지 못했습니다.")
    replacement = f"gradient_accumulation_steps={steps}"
    if match.group() == replacement:
        return None
    backup = sft_script.with_suffix(".py.before-dyson-patch")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    sft_script.write_text(source[: match.start()] + replacement + source[match.end() :], encoding="utf-8")
    return backup


def _execute(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def run_training(
    config: TrainingConfig,
    *,
    active_model_dir: Path,
    run_command: Callable[[list[str]], str] = _default_command,
    torch_module=None,
    module_available: Callable[[str], bool] | None = None,
    execute: Callable[[list[str], Path], None] = _execute,
) -> Path:
    """Train inside a run directory. This function never activates a model."""

    _ = active_model_dir
    run_dir = _run_directory(config.output_root)
    preflight = check_preflight(
        config,
        run_command=run_command,
        torch_module=torch_module,
        module_available=module_available,
    )
    if not preflight.ok:
        _write_report(run_dir, status="failed", config=config, issues=preflight.issues)
        raise PreflightFailed("\n".join(preflight.issues))

    prepared_jsonl = run_dir / "train_with_codes.jsonl"
    model_output = run_dir / "model"
    try:
        patch_gradient_accumulation(config.finetuning_dir / "sft_12hz.py", config.gradient_accumulation_steps)
        execute(
            [
                sys.executable,
                "prepare_data.py",
                "--device", "cuda:0",
                "--tokenizer_model_path", "Qwen/Qwen3-TTS-Tokenizer-12Hz",
                "--input_jsonl", str(config.train_jsonl.resolve()),
                "--output_jsonl", str(prepared_jsonl.resolve()),
            ],
            config.finetuning_dir,
        )
        execute(
            [
                sys.executable,
                "sft_12hz.py",
                "--init_model_path", config.model_name,
                "--output_model_path", str(model_output.resolve()),
                "--train_jsonl", str(prepared_jsonl.resolve()),
                "--batch_size", str(config.batch_size),
                "--lr", str(config.learning_rate),
                "--num_epochs", str(config.epochs),
                "--speaker_name", config.speaker_name,
            ],
            config.finetuning_dir,
        )
    except Exception as error:
        _write_report(run_dir, status="failed", config=config, issues=(str(error),))
        raise
    checkpoint = model_output / f"checkpoint-epoch-{config.epochs - 1}"
    _write_report(run_dir, status="completed", config=config, checkpoint_path=checkpoint)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-TTS 0.6B locally on an RTX 3070.")
    parser.add_argument("--train-jsonl", type=Path, default=Path("training_data/train.jsonl"))
    parser.add_argument("--validation-jsonl", type=Path, default=Path("training_data/validation.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("training_windows/output"))
    parser.add_argument("--finetuning-dir", type=Path, required=True)
    parser.add_argument("--active-model-dir", type=Path, default=Path("data/models/fine_tuned/dyson/active"))
    args = parser.parse_args()
    config = TrainingConfig.for_rtx_3070(
        train_jsonl=args.train_jsonl,
        validation_jsonl=args.validation_jsonl,
        output_root=args.output_root,
        finetuning_dir=args.finetuning_dir,
    )
    print(run_training(config, active_model_dir=args.active_model_dir))


if __name__ == "__main__":
    main()
