"""Report and output writers for the Moral Machine analysis."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_output_dirs(output_dir: Path, figures_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)


def write_country_summary(analysis_df: pd.DataFrame, output_path: Path) -> None:
    """Write a tidy country-level summary CSV."""
    columns = [
        "Country",
        "MeanLegalityPreference",
        "MeanUtilitarianPreference",
        "Rule_of_Law_Index",
        "Individualism_Score",
        "utilitarian_obs",
        "legality_obs",
        "MinObs",
    ]
    summary = analysis_df.loc[:, columns].sort_values(
        ["MinObs", "Country"], ascending=[False, True]
    )
    summary.to_csv(output_path, index=False)


def write_run_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    """Write run metadata in JSON format."""
    metadata = dict(metadata)
    metadata["generated_at"] = datetime.utcnow().isoformat() + "Z"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, ensure_ascii=False)


def write_analysis_report(report: dict[str, Any], output_path: Path) -> None:
    """Write a Markdown summary report."""
    outlier_legality = ", ".join(report.get("outliers_legality", [])) or "None"
    outlier_util = ", ".join(report.get("outliers_utilitarian", [])) or "None"

    lines = [
        "# Moral Machine Analysis Report",
        "",
        "## Run Overview",
        f"- Timestamp: {report['timestamp']}",
        f"- Rows scanned: {report['rows_scanned']:,}",
        f"- Countries seen in raw data: {report['countries_seen']}",
        f"- Countries after merge: {report['countries_after_merge']}",
        f"- Countries after min-obs filter: {report['countries_after_filter']}",
        f"- Minimum observations threshold: {report['min_obs']}",
        "",
        "## Hypothesis Tests",
        "### Rule of Law vs. Legality Preference",
        (
            f"- Pearson r = {report['correlations']['rule_of_law_vs_legality']['pearson_r']:.3f}, "
            f"p = {report['correlations']['rule_of_law_vs_legality']['pearson_p']:.4f}"
        ),
        (
            f"- Spearman r = {report['correlations']['rule_of_law_vs_legality']['spearman_r']:.3f}, "
            f"p = {report['correlations']['rule_of_law_vs_legality']['spearman_p']:.4f}"
        ),
        "### Individualism vs. Utilitarian Preference",
        (
            f"- Pearson r = {report['correlations']['individualism_vs_utilitarian']['pearson_r']:.3f}, "
            f"p = {report['correlations']['individualism_vs_utilitarian']['pearson_p']:.4f}"
        ),
        (
            f"- Spearman r = {report['correlations']['individualism_vs_utilitarian']['spearman_r']:.3f}, "
            f"p = {report['correlations']['individualism_vs_utilitarian']['spearman_p']:.4f}"
        ),
        "",
        "## Notable Outliers",
        f"- Legality vs. Rule of Law: {outlier_legality}",
        f"- Utilitarian vs. Individualism: {outlier_util}",
        "",
        "## Outputs",
        "- `output/figures/rule_of_law_vs_legality.png`",
        "- `output/figures/individualism_vs_utilitarian.png`",
        "- `output/figures/preference_distributions.png`",
        "- `output/figures/correlation_heatmap.png`",
        "- `output/figures/top_response_counts.png`",
        "- `output/country_summary.csv`",
        "- `output/merged_analysis.csv`",
        "- `output/run_metadata.json`",
        "",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
