from __future__ import annotations

from pathlib import Path
from typing import Callable

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from voice_tts.config import ProjectPaths, Settings
from voice_tts.schemas import PrepareReferenceRequest, SynthesisRequest
from voice_tts.services.reference import AudioReferencePreparer, ReferencePreparer, ReferenceStore
from voice_tts.services.jobs import SynthesisError, SynthesisService
from voice_tts.services.tts import TtsEngine, preferred_engine


def create_app(paths: ProjectPaths, settings: Settings, engine: TtsEngine | None = None, preparer: ReferencePreparer | None = None, concatenate: Callable[[list[Path], Path], None] | None = None) -> FastAPI:
    app = FastAPI()
    store = ReferenceStore(paths, preparer or AudioReferencePreparer())
    selected_engine, model_status = preferred_engine(paths)
    synthesis_engine = engine or selected_engine
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/api/model")
    def model() -> dict[str, str]:
        return model_status if engine is None else {"mode": "custom", "reason": "injected test engine"}

    @app.get("/api/reference/sources")
    def sources() -> dict[str, list[str]]:
        if not paths.study_source_dir.exists(): return {"sources": []}
        return {"sources": [path.name for path in paths.study_source_dir.iterdir() if path.is_file()]}

    @app.post("/api/reference/prepare")
    def prepare(request: PrepareReferenceRequest) -> dict[str, str]:
        try:
            ref = store.prepare(request.source_name, request.start_seconds, request.duration_seconds, request.transcript)
            return {"transcript": ref.transcript}
        except FileNotFoundError as error: raise HTTPException(404, "참조 녹음 파일을 찾을 수 없습니다.") from error
        except ValueError as error: raise HTTPException(422, str(error)) from error
        except Exception as error: raise HTTPException(503, str(error)) from error

    @app.post("/api/synthesis")
    def synthesize(request: SynthesisRequest) -> dict[str, str]:
        try:
            service = SynthesisService(paths, store.load(), synthesis_engine, settings.max_chunk_chars, concatenate or __import__("voice_tts.services.audio", fromlist=["concatenate_wavs"]).concatenate_wavs)
            result = service.generate(request.text)
            return {"output_id": result.output_id, "download_url": f"/api/outputs/{result.output_id}"}
        except FileNotFoundError as error: raise HTTPException(422, str(error)) from error
        except (ValueError, SynthesisError) as error: raise HTTPException(422, str(error)) from error

    @app.get("/api/outputs/{output_id}")
    def output(output_id: str) -> FileResponse:
        if not output_id.isalnum(): raise HTTPException(404, "출력 파일을 찾을 수 없습니다.")
        path = paths.output_dir / f"{output_id}.wav"
        if not path.is_file(): raise HTTPException(404, "출력 파일을 찾을 수 없습니다.")
        return FileResponse(path, media_type="audio/wav", filename="my-voice.wav")
    return app


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    uvicorn.run(create_app(ProjectPaths.from_root(root), Settings()), host="127.0.0.1", port=8765)
