"""
Moral Machine analysis runner.

This script orchestrates the pipeline defined in `moral_machine_analysis`.
Edit the configuration values below to tune chunk sizes, filtering, and output paths.
"""

from pathlib import Path

from moral_machine_analysis.config import RunConfig
from moral_machine_analysis.pipeline import run_analysis

SCRIPT_DIR = Path(__file__).resolve().parent

# Adjust these settings to tune performance or change output locations.
DATA_DIR = SCRIPT_DIR / "Data"
OUTPUT_DIR = SCRIPT_DIR / "output"
CHUNK_ROWS = 1_000_000
MIN_COUNTRY_OBS = 100
OUTLIER_LABELS = 8
TOP_COUNTRIES = 15
SHOW_PLOTS = False


def main() -> None:
    config = RunConfig(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        chunk_rows=CHUNK_ROWS,
        min_obs=MIN_COUNTRY_OBS,
        outlier_labels=OUTLIER_LABELS,
        top_countries=TOP_COUNTRIES,
        show_plots=SHOW_PLOTS,
    )
    run_analysis(config)


if __name__ == "__main__":
    main()
