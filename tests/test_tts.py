import sys
from pathlib import Path
from types import ModuleType

from voice_tts.config import ProjectPaths
from voice_tts.services.tts import FineTunedQwenTtsEngine, QwenTtsEngine, preferred_engine


def test_qwen_adapter_sends_korean_to_voice_clone_and_writes_wav(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, str]] = []
    download_calls: list[tuple[str, str]] = []

    class FakeQwenModel:
        instances = 0

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: str):
            assert model_name == str(tmp_path / "model")
            cls.instances += 1
            return cls()

        def generate_voice_clone(self, **kwargs: str):
            calls.append(kwargs)
            return [[0.0, 0.1]], 24000

    qwen_tts = ModuleType("qwen_tts")
    qwen_tts.Qwen3TTSModel = FakeQwenModel
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_tts)
    monkeypatch.setenv("HF_HOME", "/an-unrelated-cache")

    huggingface_hub = ModuleType("huggingface_hub")

    def snapshot_download(model_name: str, *, cache_dir: str) -> str:
        download_calls.append((model_name, cache_dir))
        return str(tmp_path / "model")

    huggingface_hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)

    soundfile = ModuleType("soundfile")
    soundfile.write = lambda path, wav, sample_rate: Path(path).write_bytes(b"wav")
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    engine = QwenTtsEngine(cache_dir=tmp_path)

    output = tmp_path / "out.wav"
    engine.synthesize(Path("ref.wav"), "참조 대본", "안녕하세요.", output)
    engine.synthesize(Path("ref.wav"), "참조 대본", "다른 문장", tmp_path / "other.wav")

    assert FakeQwenModel.instances == 1
    assert download_calls == [("Qwen/Qwen3-TTS-12Hz-0.6B-Base", str(tmp_path))]
    assert calls[0]["language"] == "Korean"
    assert calls[0]["ref_audio"] == "ref.wav"
    assert calls[0]["ref_text"] == "참조 대본"
    assert calls[0]["text"] == "안녕하세요."
    assert __import__("os").environ["HF_HOME"] == str(tmp_path)
    assert output.read_bytes() == b"wav"


def test_preferred_engine_uses_valid_active_custom_voice_model(tmp_path: Path, monkeypatch) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    active = paths.fine_tuned_model_dir / "active"
    active.mkdir(parents=True)
    (active / "model.safetensors").write_bytes(b"weights")
    (active / "model_manifest.json").write_text('{"speaker":"dyson","language":"Korean"}', encoding="utf-8")

    engine, status = preferred_engine(paths)

    assert isinstance(engine, FineTunedQwenTtsEngine)
    assert status["mode"] == "fine_tuned"


def test_fine_tuned_engine_calls_custom_voice_in_korean(tmp_path: Path, monkeypatch) -> None:
    calls = []
    class FakeModel:
        @classmethod
        def from_pretrained(cls, path: str): return cls()
        def generate_custom_voice(self, **kwargs): calls.append(kwargs); return [[0.0]], 24000
    module = ModuleType("qwen_tts"); module.Qwen3TTSModel = FakeModel; monkeypatch.setitem(sys.modules, "qwen_tts", module)
    soundfile = ModuleType("soundfile"); soundfile.write = lambda path, wav, rate: Path(path).write_bytes(b"wav"); monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    engine = FineTunedQwenTtsEngine(tmp_path)
    engine.synthesize(Path("unused.wav"), "unused", "안녕하세요.", tmp_path / "out.wav")
    assert calls == [{"text": "안녕하세요.", "language": "Korean", "speaker": "dyson"}]
