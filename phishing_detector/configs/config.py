from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    base_dir: Path
    data_dir: Path
    datasets_dir: Path
    models_dir: Path
    logs_dir: Path
    seed: int = 42


def get_config(base_dir: str | None = None) -> ProjectConfig:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]
    else:
        base_dir = Path(base_dir).resolve()

    return ProjectConfig(
        base_dir=base_dir,
        data_dir=base_dir / "phishing_detector/data",
        datasets_dir=base_dir / "phishing_detector/datasets",
        models_dir=base_dir / "phishing_detector/models",
        logs_dir=base_dir / "phishing_detector/logs",
    )
