# Moral Machine Analysis (COG260)
This repository showcases large-scale CSV processing with Python, feature engineering, statistical testing, and reproducible outputs. The pipeline analyzes Moral Machine decisions at country scale and links them to cultural indicators (rule of law and individualism).

## Data
Download the Moral Machine dataset here:
https://osf.io/3hvt2/overview?view_only=4bb49492edee4a8eb1758552a362a2cf

From the download, extract `SharedResponses.csv` (inside `SharedResponses.csv.tar.gz`) and place it in `Data/` alongside:
- `RuleOfLaw.csv`
- `IndividualisticRanking.csv`

## Setup
Install the requirements:
```
pip install -r requirements.txt
```

## Run
```
python Project_PythonScript.py
```

Adjust `Project_PythonScript.py` if you want to change chunk sizes, minimum observations, or output paths.

## Outputs (reproducible)
Generated artifacts are written to `output/`:
- `analysis_report.md` (summary of key results)
- `run_metadata.json` (run config, dataset sizes, version info)
- `country_summary.csv` (cleaned metrics per country)
- `merged_analysis.csv` (full merged dataset)
- `figures/` (high-resolution visuals)

## Highlights
- Chunked processing for multi-GB CSVs
- Country name normalization + schema aliasing
- Hypothesis tests (Pearson + Spearman correlations)
- Aesthetic, high-contrast visuals for portfolio-ready plots
