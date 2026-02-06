# Moral Machine Analysis (COG260)

A data-driven investigation into how **cultural values shape moral decisions** in autonomous-vehicle dilemmas. This project processes millions of rows from MIT's Moral Machine experiment, engineers country-level preference metrics, and tests two hypotheses linking national culture to ethical choices using statistical correlation analysis.

---

## Table of Contents

- [Overview](#overview)
- [Research Questions](#research-questions)
- [Results](#results)
  - [Hypothesis 1 -- Rule of Law vs. Legality Preference](#hypothesis-1----rule-of-law-vs-legality-preference)
  - [Hypothesis 2 -- Individualism vs. Utilitarian Preference](#hypothesis-2----individualism-vs-utilitarian-preference)
  - [Preference Distributions](#preference-distributions)
  - [Correlation Heatmap](#correlation-heatmap)
  - [Top Countries by Response Volume](#top-countries-by-response-volume)
- [Technical Highlights](#technical-highlights)
- [Data](#data)
- [Setup](#setup)
- [Run](#run)
- [Outputs](#outputs)
- [Project Structure](#project-structure)

---

## Overview

The [Moral Machine](https://www.moralmachine.net/) experiment (Awad et al., 2018) collected tens of millions of human judgements on ethical dilemmas facing autonomous vehicles. Each scenario forces a binary choice -- for example, swerving to save pedestrians at the cost of passengers, or obeying traffic laws versus saving more lives.

This project builds a **fully automated, reproducible pipeline** that:

1. **Streams** the multi-GB Moral Machine CSV in memory-efficient chunks
2. **Engineers** two behavioural metrics per country -- *Legality Preference* and *Utilitarian Preference*
3. **Merges** country-level preferences with external cultural indicators (Rule of Law Index, Hofstede Individualism Score)
4. **Tests** two hypotheses with Pearson and Spearman correlations plus linear regression
5. **Generates** publication-quality visualisations and a structured analysis report

---

## Research Questions

| # | Hypothesis | Independent Variable | Dependent Variable |
|---|-----------|----------------------|-------------------|
| H1 | Countries with stronger rule of law show greater preference for legal (law-abiding) outcomes in moral dilemmas | Rule of Law Index | Mean Legality Preference |
| H2 | More individualistic countries show greater preference for utilitarian (save-more-lives) outcomes | Hofstede Individualism Score | Mean Utilitarian Preference |

---

## Results

### Hypothesis 1 -- Rule of Law vs. Legality Preference

A scatter plot with a best-fit regression line reveals the relationship between a country's Rule of Law Index and its residents' average preference for the law-abiding choice in pedestrian-vs-pedestrian traffic-light dilemmas. Outlier countries with the largest residuals from the trend line are annotated.

> **Key finding:** Pearson and Spearman correlations are computed to quantify the strength and direction of the relationship. The regression line slope, p-value, and confidence are all reported in the generated analysis report.

<div align="center">
  <img width="auto" height="400" alt="rule_of_law_vs_legality" src="https://github.com/user-attachments/assets/708950d7-a819-4ec3-8e13-fdeb317f2e44" />
</div>

---

### Hypothesis 2 -- Individualism vs. Utilitarian Preference

This scatter plot examines whether countries scoring higher on Hofstede's Individualism dimension tend to favour saving the greater number of lives (utilitarian choice). A regression line is overlaid and notable outliers are labelled.

> **Key finding:** Both parametric (Pearson) and non-parametric (Spearman) tests are run. The analysis report captures the correlation coefficients and their statistical significance.

<div align="center">
  <img width="auto" height="400" alt="individualism_vs_utilitarian" src="https://github.com/user-attachments/assets/f81c08cc-6858-4800-8be2-69902f121bf3" />
</div>

---

### Preference Distributions

Side-by-side histograms with KDE (Kernel Density Estimation as opposed to Probablity) overlays show the cross-country distribution of Legality Preference and Utilitarian Preference. The dashed vertical line marks the global mean for each metric, making it easy to see skew and spread.

<div align="center">
  <img width="auto" height="400" alt="individualism_vs_utilitarian" src="https://github.com/user-attachments/assets/91bd04a7-2b37-4c08-93ea-c5f0ecd3d2b7" />
</div>

---

### Correlation Heatmap

A four-variable correlation matrix (Rule of Law Index, Individualism Score, Mean Legality Preference, Mean Utilitarian Preference) visualised as an annotated heatmap. This provides a single-glance summary of all pairwise relationships, helping identify multicollinearity or unexpected associations.

<div align="center">
 <img width="auto" height="400" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/7e222e86-4249-4399-a4b6-0692a547299e" />
</div>

---

### Top Countries by Response Volume

A horizontal bar chart of the 15 countries with the most Moral Machine responses (after filtering). This contextualises the statistical results -- countries with larger sample sizes carry more weight and their estimates are more stable.



<div align="center">
 <img width="auto" height="400" alt="top_response_counts" src="https://github.com/user-attachments/assets/e672ed1f-5e62-41d5-8a64-08c87f1e5b16" />
</div>

---

## Technical Highlights

| Capability | Detail |
|-----------|--------|
| **Large-scale data processing** | Streams multi-GB CSVs in configurable chunks (default 1 M rows) to stay within memory limits |
| **Feature engineering** | Derives Legality and Utilitarian preference scores from raw scenario-level binary choices |
| **Country normalisation** | Handles alternate names, casing, and schema aliases across three independent datasets |
| **Hypothesis testing** | Pearson correlation, Spearman rank correlation, and OLS linear regression with full diagnostics |
| **Outlier detection** | Identifies countries with the largest absolute residuals from the best-fit line |
| **Reproducible outputs** | Every run writes timestamped metadata (Python version, pandas version, platform, config) to JSON |
| **Publication-quality plots** | Dark-theme, high-contrast figures saved at 300 DPI, ready for reports or portfolios |
| **Modular architecture** | Clean separation into `config`, `data_io`, `metrics`, `plots`, `reporting`, and `pipeline` modules |

---

## Data

Download the Moral Machine dataset here:
https://osf.io/3hvt2/overview?view_only=4bb49492edee4a8eb1758552a362a2cf

From the download, extract `SharedResponses.csv` (inside `SharedResponses.csv.tar.gz`) and place it in `Data/` alongside the two cultural-indicator files already included in this repo:

```
Data/
├── SharedResponses.csv          # ~500 MB+ (not committed)
├── RuleOfLaw.csv                # World Bank Rule of Law Index
└── IndividualisticRanking.csv   # Hofstede Individualism Scores
```

---

## Setup

```bash
pip install -r requirements.txt
```

Core dependencies: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `pyarrow` (optional, for faster I/O).

---

## Run

```bash
python Project_PythonScript.py
```

Adjust `Project_PythonScript.py` if you want to change chunk sizes, minimum observation thresholds, or output paths.

---

## Outputs

All generated artifacts are written to `output/` (gitignored for clean repo history):

| File | Description |
|------|-------------|
| `analysis_report.md` | Markdown summary with correlation results, outlier lists, and output manifest |
| `run_metadata.json` | Full run config, dataset sizes, library versions, and statistical results |
| `country_summary.csv` | Tidy country-level table of preferences, cultural scores, and observation counts |
| `merged_analysis.csv` | Complete merged dataset used for all analyses |
| `figures/rule_of_law_vs_legality.png` | Scatter + regression: Rule of Law vs. Legality Preference |
| `figures/individualism_vs_utilitarian.png` | Scatter + regression: Individualism vs. Utilitarian Preference |
| `figures/preference_distributions.png` | Histograms of both preference metrics |
| `figures/correlation_heatmap.png` | Annotated 4-variable correlation matrix |
| `figures/top_response_counts.png` | Bar chart of top 15 countries by response volume |

---

## Project Structure

```
COG260-Project-Moral-Machine-Analysis/
├── Data/
│   ├── IndividualisticRanking.csv
│   └── RuleOfLaw.csv
├── moral_machine_analysis/        # Modular analysis package
│   ├── __init__.py
│   ├── config.py                  # RunConfig dataclass
│   ├── data_io.py                 # CSV loading, country normalisation
│   ├── metrics.py                 # Aggregation, correlations, regression
│   ├── plots.py                   # All visualisation functions
│   ├── pipeline.py                # End-to-end orchestration
│   └── reporting.py               # Report & metadata writers
├── Project_PythonScript.py        # Entry point
├── requirements.txt
└── README.md
```
