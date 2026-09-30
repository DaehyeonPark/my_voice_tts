"""Generate fixed Korean holdouts for local listening review."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


def load_holdouts(path: Path) -> list[str]:
    sentences = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(sentences) != 5 or len(set(sentences)) != 5:
        raise ValueError("홀드아웃은 중복 없는 한국어 문장 5개여야 합니다.")
    return sentences


def evaluate_checkpoint(checkpoint: Path, output_dir: Path, holdout_path: Path, *, listening_approved: bool) -> Path:
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    sentences = load_holdouts(holdout_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = Qwen3TTSModel.from_pretrained(
        str(checkpoint), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    outputs: list[dict[str, object]] = []
    for index, text in enumerate(sentences, start=1):
        wavs, sample_rate = model.generate_custom_voice(text=text, language="Korean", speaker="dyson")
        if not wavs or sample_rate <= 0:
            raise RuntimeError(f"홀드아웃 {index} 생성에 실패했습니다.")
        wav_path = output_dir / f"holdout-{index:02d}.wav"
        sf.write(wav_path, wavs[0], sample_rate)
        info = sf.info(wav_path)
        if not wav_path.is_file() or info.samplerate != sample_rate:
            raise RuntimeError(f"홀드아웃 {index} WAV 검증에 실패했습니다.")
        outputs.append({"text": text, "audio": str(wav_path), "sample_rate": sample_rate})
    report = {
        "status": "passed" if listening_approved else "review_required",
        "automatic_checks_passed": True,
        "listening_approved": listening_approved,
        "speaker": "dyson",
        "language": "Korean",
        "checkpoint": str(checkpoint.resolve()),
        "model_sha256": sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest(),
        "outputs": outputs,
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate five Korean holdouts for listening review.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdouts", type=Path, default=Path("training_windows/holdout_ko.txt"))
    parser.add_argument("--listening-approved", action="store_true")
    args = parser.parse_args()
    print(evaluate_checkpoint(args.checkpoint, args.output_dir, args.holdouts, listening_approved=args.listening_approved))


if __name__ == "__main__":
    main()
