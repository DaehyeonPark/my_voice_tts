from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from voice_tts.app import create_app
from voice_tts.config import ProjectPaths, Settings


class FakePreparer:
    def prepare_reference(self, source: Path, start: float, duration: float, destination: Path) -> None:
        destination.write_bytes(b"reference")


class FakeEngine:
    def synthesize(self, reference_audio: Path, reference_text: str, text: str, output_path: Path) -> None:
        output_path.write_bytes(b"generated")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    paths = ProjectPaths.from_root(tmp_path)
    paths.study_source_dir.mkdir()
    (paths.study_source_dir / "voice.m4a").write_bytes(b"source")
    return TestClient(create_app(paths, Settings(), engine=FakeEngine(), preparer=FakePreparer(), concatenate=lambda files, destination: destination.write_bytes(b"generated")))


def test_prepare_synthesis_and_download_contract(client: TestClient) -> None:
    prepared = client.post("/api/reference/prepare", json={"source_name": "voice.m4a", "start_seconds": 0, "duration_seconds": 15, "transcript": "안녕하세요."})
    assert prepared.status_code == 200
    generated = client.post("/api/synthesis", json={"text": "테스트 문장입니다."})
    assert generated.status_code == 200
    assert generated.json()["download_url"].startswith("/api/outputs/")


def test_rejects_path_traversal_and_blank_transcript(client: TestClient) -> None:
    assert client.post("/api/reference/prepare", json={"source_name": "../secret.m4a", "start_seconds": 0, "duration_seconds": 15, "transcript": "대본"}).status_code == 404
    assert client.post("/api/reference/prepare", json={"source_name": "voice.m4a", "start_seconds": 0, "duration_seconds": 15, "transcript": " "}).status_code == 422


def test_home_page_exposes_local_voice_workflow(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "참조 음성" in response.text
    assert "WAV 다운로드" in response.text
    assert "127.0.0.1에서만" in response.text
