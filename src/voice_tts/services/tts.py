"""Model adapter for local Qwen3-TTS Korean voice-clone inference."""

from __future__ import annotations

import os
import json
from pathlib import Path
from threading import Lock
from typing import Protocol

from voice_tts.config import ProjectPaths
from voice_tts.runtime import model_load_options


class SynthesisError(RuntimeError):
    """Raised when local speech synthesis cannot finish."""


class TtsEngine(Protocol):
    def synthesize(self, reference_audio: Path, reference_text: str, text: str, output_path: Path) -> None: ...


class QwenTtsEngine:
    """Lazy Qwen3-TTS adapter that always requests Korean synthesis."""

    model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._model = None
        self._lock = Lock()

    def synthesize(self, reference_audio: Path, reference_text: str, text: str, output_path: Path) -> None:
        model = self._load_model()
        try:
            import soundfile

            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language="Korean",
                ref_audio=str(reference_audio),
                ref_text=reference_text,
            )
            soundfile.write(str(output_path), wavs[0], sample_rate)
        except Exception as error:
            raise SynthesisError("Qwen3-TTS가 한국어 음성을 생성하지 못했습니다.") from error

    def _load_model(self):
        with self._lock:
            if self._model is None:
                try:
                    from qwen_tts import Qwen3TTSModel
                    from huggingface_hub import snapshot_download

                    self._cache_dir.mkdir(parents=True, exist_ok=True)
                    os.environ["HF_HOME"] = str(self._cache_dir)
                    model_path = snapshot_download(self.model_name, cache_dir=str(self._cache_dir))
                    self._model = Qwen3TTSModel.from_pretrained(model_path, **model_load_options())
                except Exception as error:
                    raise SynthesisError("Qwen3-TTS 한국어 모델을 준비하지 못했습니다. 인터넷 연결과 설치 상태를 확인해 주세요.") from error
            return self._model


class FineTunedQwenTtsEngine:
    """CustomVoice inference for an activated, locally fine-tuned model."""
    def __init__(self, model_dir: Path, speaker: str = "dyson", *, fallback: TtsEngine | None = None, status: dict[str, str] | None = None) -> None:
        self._model_dir, self._speaker, self._model, self._lock = model_dir, speaker, None, Lock()
        self._fallback, self._status, self._load_failed = fallback, status, False

    def synthesize(self, reference_audio: Path, reference_text: str, text: str, output_path: Path) -> None:
        if not self._load_failed:
            try:
                model = self._load_model()
            except Exception as error:
                if self._fallback is None:
                    raise SynthesisError("학습된 한국어 모델을 로드하지 못했습니다.") from error
                self._load_failed = True
                if self._status is not None:
                    self._status.update(mode="zero_shot", reason="active model load failed; using base voice clone")
        if self._load_failed:
            return self._fallback.synthesize(reference_audio, reference_text, text, output_path)
        try:
            import soundfile
            wavs, sample_rate = model.generate_custom_voice(text=text, language="Korean", speaker=self._speaker)
            soundfile.write(str(output_path), wavs[0], sample_rate)
        except Exception as error:
            raise SynthesisError("학습된 한국어 음성을 생성하지 못했습니다.") from error

    def _load_model(self):
        with self._lock:
            if self._model is None:
                from qwen_tts import Qwen3TTSModel
                self._model = Qwen3TTSModel.from_pretrained(str(self._model_dir), **model_load_options())
            return self._model


def preferred_engine(paths: ProjectPaths) -> tuple[TtsEngine, dict[str, str]]:
    active = paths.fine_tuned_model_dir / "active"
    manifest = active / "model_manifest.json"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("speaker") == "dyson" and data.get("language") == "Korean" and (active / "model.safetensors").is_file():
            status = {"mode": "fine_tuned", "reason": "active model selected; load pending"}
            return FineTunedQwenTtsEngine(active, fallback=QwenTtsEngine(paths.model_dir), status=status), status
    except (OSError, json.JSONDecodeError):
        pass
    return QwenTtsEngine(paths.model_dir), {"mode": "zero_shot", "reason": "no valid active fine-tuned model"}
