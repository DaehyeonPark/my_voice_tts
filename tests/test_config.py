from pathlib import Path

from voice_tts.config import ProjectPaths, Settings


def test_runtime_creation_does_not_create_or_modify_study_source(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)

    paths.ensure_runtime_directories()

    assert paths.reference_dir.is_dir()
    assert paths.output_dir.is_dir()
    assert not paths.study_source_dir.exists()


def test_settings_are_loopback_only() -> None:
    assert Settings().host == "127.0.0.1"


def test_main_function_is_available_for_the_local_run_command() -> None:
    from importlib.metadata import entry_points

    commands = {entry.name: entry.value for entry in entry_points(group="console_scripts")}
    assert commands["voice-tts"] == "voice_tts.app:main"
