"""End-to-end pipeline for the Moral Machine analysis."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import platform
import sys

import pandas as pd

from .config import FIGURES_DIRNAME, RunConfig
from .data_io import INDIVIDUALISM_ALIASES, RULE_OF_LAW_ALIASES, load_lookup_table
from .metrics import (
    aggregate_moral_preferences,
    compute_correlations,
    find_outlier_labels,
    fit_linear_regression,
)
from .plots import (
    plot_correlation_heatmap,
    plot_legality_vs_rule_of_law,
    plot_preference_distributions,
    plot_top_response_counts,
    plot_utilitarian_vs_individualism,
    set_plot_style,
)
from .reporting import ensure_output_dirs, write_analysis_report, write_country_summary, write_run_metadata


def _validate_inputs(data_dir: Path) -> dict[str, Path]:
    moral_machine_path = data_dir / "SharedResponses.csv"
    rule_of_law_path = data_dir / "RuleOfLaw.csv"
    individualism_path = data_dir / "IndividualisticRanking.csv"
    for path in (moral_machine_path, rule_of_law_path, individualism_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    return {
        "moral_machine": moral_machine_path,
        "rule_of_law": rule_of_law_path,
        "individualism": individualism_path,
    }


def _file_metadata(path: Path) -> dict[str, str | float]:
    stat = path.stat()
    return {
        "size_mb": round(stat.st_size / 1_048_576, 2),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def run_analysis(config: RunConfig) -> pd.DataFrame:
    """Run the full analysis pipeline and return the final analysis DataFrame."""
    input_paths = _validate_inputs(config.data_dir)
    output_dir = config.output_dir
    figures_dir = output_dir / FIGURES_DIRNAME
    ensure_output_dirs(output_dir, figures_dir)
    set_plot_style()

    print(f"Using data directory: {config.data_dir}")
    for label, path in input_paths.items():
        meta = _file_metadata(path)
        print(f" - {label}: {path.name} ({meta['size_mb']} MB)")

    rule_of_law_df = load_lookup_table(
        input_paths["rule_of_law"], RULE_OF_LAW_ALIASES
    )
    individualism_df = load_lookup_table(
        input_paths["individualism"], INDIVIDUALISM_ALIASES
    )

    moral_country_means, agg_meta = aggregate_moral_preferences(
        input_paths["moral_machine"], chunk_rows=config.chunk_rows
    )
    print(
        f"Processed {agg_meta['rows_scanned']:,} rows across "
        f"{agg_meta['countries_seen']} countries."
    )

    merged = (
        moral_country_means.reset_index()
        .merge(rule_of_law_df, on="Country", how="inner")
        .merge(individualism_df, on="Country", how="inner")
        .dropna()
    )

    analysis_df = merged.rename(
        columns={
            "Mean_Utilitarian": "MeanUtilitarianPreference",
            "Mean_Legality": "MeanLegalityPreference",
        }
    )
    analysis_df["MinObs"] = analysis_df[["utilitarian_obs", "legality_obs"]].min(axis=1)
    analysis_df = analysis_df[analysis_df["MinObs"] >= config.min_obs].copy()
    analysis_df = analysis_df.sort_values("Country").reset_index(drop=True)
    if analysis_df.empty:
        raise ValueError(
            "No countries remain after filtering. Lower MIN_COUNTRY_OBS and rerun."
        )

    correlations = compute_correlations(analysis_df)
    regressions = {
        "rule_of_law_vs_legality": fit_linear_regression(
            analysis_df, "Rule_of_Law_Index", "MeanLegalityPreference"
        ),
        "individualism_vs_utilitarian": fit_linear_regression(
            analysis_df, "Individualism_Score", "MeanUtilitarianPreference"
        ),
    }

    legality_outliers = (
        find_outlier_labels(
            analysis_df,
            "Rule_of_Law_Index",
            "MeanLegalityPreference",
            config.outlier_labels,
        )
        if len(analysis_df) >= 2
        else analysis_df.head(0)
    )
    util_outliers = None
    if len(analysis_df) >= 2 and analysis_df["MeanUtilitarianPreference"].nunique() > 1:
        util_outliers = find_outlier_labels(
            analysis_df,
            "Individualism_Score",
            "MeanUtilitarianPreference",
            config.outlier_labels,
        )

    plot_legality_vs_rule_of_law(
        analysis_df,
        correlations["rule_of_law_vs_legality"],
        legality_outliers,
        figures_dir / "rule_of_law_vs_legality.png",
        show_plot=config.show_plots,
    )
    plot_utilitarian_vs_individualism(
        analysis_df,
        correlations["individualism_vs_utilitarian"],
        util_outliers,
        figures_dir / "individualism_vs_utilitarian.png",
        show_plot=config.show_plots,
    )
    plot_preference_distributions(
        analysis_df,
        figures_dir / "preference_distributions.png",
        show_plot=config.show_plots,
    )
    plot_correlation_heatmap(
        analysis_df,
        figures_dir / "correlation_heatmap.png",
        show_plot=config.show_plots,
    )
    plot_top_response_counts(
        analysis_df,
        figures_dir / "top_response_counts.png",
        top_n=config.top_countries,
        show_plot=config.show_plots,
    )

    analysis_df.to_csv(output_dir / "merged_analysis.csv", index=False)
    write_country_summary(analysis_df, output_dir / "country_summary.csv")

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "rows_scanned": agg_meta["rows_scanned"],
        "countries_seen": agg_meta["countries_seen"],
        "countries_after_merge": int(len(merged)),
        "countries_after_filter": int(len(analysis_df)),
        "min_obs": config.min_obs,
        "correlations": correlations,
        "outliers_legality": legality_outliers["Country"].tolist(),
        "outliers_utilitarian": util_outliers["Country"].tolist()
        if util_outliers is not None
        else [],
    }
    write_analysis_report(report, output_dir / "analysis_report.md")

    metadata = {
        "config": {
            "data_dir": str(config.data_dir),
            "output_dir": str(config.output_dir),
            "chunk_rows": config.chunk_rows,
            "min_obs": config.min_obs,
            "outlier_labels": config.outlier_labels,
            "top_countries": config.top_countries,
            "seed": config.seed,
        },
        "input_files": {name: _file_metadata(path) for name, path in input_paths.items()},
        "correlations": correlations,
        "regressions": regressions,
        "rows_scanned": agg_meta["rows_scanned"],
        "countries_seen": agg_meta["countries_seen"],
        "countries_after_merge": int(len(merged)),
        "countries_after_filter": int(len(analysis_df)),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
    }
    write_run_metadata(metadata, output_dir / "run_metadata.json")

    print("Analysis complete. Outputs written to:")
    print(f" - {output_dir / 'analysis_report.md'}")
    print(f" - {output_dir / 'country_summary.csv'}")
    print(f" - {output_dir / 'merged_analysis.csv'}")
    print(f" - {figures_dir}")

    return analysis_df
