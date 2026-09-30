"""Fixed local runtime configuration and project-owned paths."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    reference_sample_rate: int = 24_000
    reference_channels: int = 1
    max_chunk_chars: int = 260


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    study_source_dir: Path
    data_dir: Path
    reference_dir: Path
    output_dir: Path
    model_dir: Path
    fine_tuned_model_dir: Path
    temp_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        project_root = root.resolve()
        data_dir = project_root / "data"
        return cls(
            root=project_root,
            study_source_dir=project_root / "study_src",
            data_dir=data_dir,
            reference_dir=data_dir / "reference",
            output_dir=data_dir / "outputs",
            model_dir=data_dir / "models",
            fine_tuned_model_dir=data_dir / "models" / "fine_tuned" / "dyson",
            temp_dir=project_root / "tmp",
        )

    def ensure_runtime_directories(self) -> None:
        for directory in (self.reference_dir, self.output_dir, self.model_dir, self.fine_tuned_model_dir, self.temp_dir):
            directory.mkdir(parents=True, exist_ok=True)
