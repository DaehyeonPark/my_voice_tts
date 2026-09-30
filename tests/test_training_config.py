import json
from pathlib import Path

import pytest

from training_windows.train_0_6b import (
    PreflightFailed,
    TrainingConfig,
    check_preflight,
    patch_gradient_accumulation,
    run_training,
)


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(_: int):
            return type("Properties", (), {"total_memory": 8 * 1024**3})()


def make_config(tmp_path: Path) -> TrainingConfig:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text('{"audio":"a.wav"}\n', encoding="utf-8")
    validation.write_text('{"audio":"b.wav"}\n', encoding="utf-8")
    finetuning = tmp_path / "Qwen3-TTS" / "finetuning"
    finetuning.mkdir(parents=True)
    (finetuning / "prepare_data.py").write_text("# prepare\n", encoding="utf-8")
    (finetuning / "sft_12hz.py").write_text(
        "accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision='bf16')\n",
        encoding="utf-8",
    )
    return TrainingConfig.for_rtx_3070(
        train_jsonl=train,
        validation_jsonl=validation,
        output_root=tmp_path / "output",
        finetuning_dir=finetuning,
    )


def test_rtx_3070_configuration_uses_the_approved_small_model_settings(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    assert config.model_name == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    assert config.batch_size == 1
    assert config.gradient_accumulation_steps == 8
    assert config.epochs == 3
    assert config.learning_rate == 2e-5
    assert config.speaker_name == "dyson"
    assert config.language == "Korean"


def test_preflight_accepts_cuda_eight_gb_flash_attention_and_nonempty_datasets(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    result = check_preflight(
        config,
        run_command=lambda _: "8192\n",
        torch_module=FakeTorch,
        module_available=lambda _: True,
    )

    assert result.ok is True
    assert result.issues == ()


def test_failed_preflight_writes_failed_report_without_changing_active_model(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.train_jsonl.unlink()
    active_model = tmp_path / "active"
    active_model.mkdir()
    marker = active_model / "model_manifest.json"
    marker.write_text('{"version":"old"}', encoding="utf-8")

    with pytest.raises(PreflightFailed):
        run_training(
            config,
            active_model_dir=active_model,
            run_command=lambda _: "0\n",
            torch_module=FakeTorch,
            module_available=lambda _: False,
        )

    report = json.loads(next(config.output_root.glob("run-*/training_report.json")).read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert marker.read_text(encoding="utf-8") == '{"version":"old"}'


def test_gradient_accumulation_patch_keeps_a_backup_of_the_official_script(tmp_path: Path) -> None:
    script = tmp_path / "sft_12hz.py"
    script.write_text("Accelerator(gradient_accumulation_steps=4)\n", encoding="utf-8")

    backup = patch_gradient_accumulation(script, 8)

    assert backup is not None
    assert backup.read_text(encoding="utf-8") == "Accelerator(gradient_accumulation_steps=4)\n"
    assert "gradient_accumulation_steps=8" in script.read_text(encoding="utf-8")
