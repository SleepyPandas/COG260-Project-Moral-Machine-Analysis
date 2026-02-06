"""Configuration for the Moral Machine analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIRNAME = "figures"


@dataclass(frozen=True)
class RunConfig:
    """Run-time configuration for the analysis pipeline."""

    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    chunk_rows: int = 500_000
    min_obs: int = 100
    outlier_labels: int = 8
    top_countries: int = 15
    seed: int = 42
    show_plots: bool = False
