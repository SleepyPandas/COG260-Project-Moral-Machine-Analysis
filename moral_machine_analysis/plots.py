"""Plotting helpers for the Moral Machine analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BACKGROUND = "#0b0f1a"
AXES_BG = "#0f172a"
GRID_COLOR = "#23304f"
TEXT_COLOR = "#e5e7eb"
SPINE_COLOR = "#334155"
SCATTER_COLOR = "#2ca9c0"
BEST_FIT_COLOR = "#c23b22"
HIGHLIGHT_COLOR = "#f6c85f"
ANNOTATION_FACE_COLOR = "#111827"


def set_plot_style() -> None:
    """Configure a high-contrast dark theme for all plots."""
    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid", context="talk", font="DejaVu Sans")
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": AXES_BG,
            "axes.edgecolor": SPINE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "text.color": TEXT_COLOR,
            "savefig.facecolor": BACKGROUND,
            "savefig.edgecolor": BACKGROUND,
        }
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
    ax.tick_params(colors=TEXT_COLOR)


def _annotate_outliers(ax: plt.Axes, outlier_labels: pd.DataFrame, x_col: str, y_col: str):
    for _, row in outlier_labels.iterrows():
        ax.annotate(
            row["Country"],
            (row[x_col], row[y_col]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            color=TEXT_COLOR,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": ANNOTATION_FACE_COLOR,
                "alpha": 0.7,
                "edgecolor": BEST_FIT_COLOR,
            },
        )


def plot_legality_vs_rule_of_law(
    analysis_df: pd.DataFrame,
    corr_stats: dict[str, float],
    outlier_labels: pd.DataFrame | None,
    output_path: Path,
    show_plot: bool = False,
) -> None:
    """Create and save the legality vs. rule-of-law scatter with regression line."""
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    sns.regplot(
        ax=ax,
        data=analysis_df,
        x="Rule_of_Law_Index",
        y="MeanLegalityPreference",
        scatter_kws={"s": 70, "alpha": 0.85, "color": SCATTER_COLOR, "edgecolor": "none"},
        line_kws={"color": BEST_FIT_COLOR, "linewidth": 2.5},
        color=SCATTER_COLOR,
    )
    if outlier_labels is not None and len(outlier_labels):
        _annotate_outliers(ax, outlier_labels, "Rule_of_Law_Index", "MeanLegalityPreference")
    title = (
        "Rule of Law vs. Legality Preference\n"
        f"Pearson r = {corr_stats['pearson_r']:.2f}, p = {corr_stats['pearson_p']:.3f}"
    )
    ax.set_title(title)
    ax.set_xlabel("Rule of Law Index")
    ax.set_ylabel("Mean Legality Preference")
    _style_axes(ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_utilitarian_vs_individualism(
    analysis_df: pd.DataFrame,
    corr_stats: dict[str, float],
    outlier_labels: pd.DataFrame | None,
    output_path: Path,
    show_plot: bool = False,
) -> None:
    """Create and save the utilitarian vs. individualism scatter with regression line."""
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    if analysis_df["MeanUtilitarianPreference"].nunique() > 1:
        sns.regplot(
            ax=ax,
            data=analysis_df,
            x="Individualism_Score",
            y="MeanUtilitarianPreference",
            scatter_kws={"s": 70, "alpha": 0.85, "color": SCATTER_COLOR, "edgecolor": "none"},
            line_kws={"color": BEST_FIT_COLOR, "linewidth": 2.5},
            color=SCATTER_COLOR,
        )
        if outlier_labels is not None and len(outlier_labels):
            _annotate_outliers(
                ax, outlier_labels, "Individualism_Score", "MeanUtilitarianPreference"
            )
        title = (
            "Individualism vs. Utilitarian Preference\n"
            f"Pearson r = {corr_stats['pearson_r']:.2f}, p = {corr_stats['pearson_p']:.3f}"
        )
    else:
        sns.scatterplot(
            ax=ax,
            data=analysis_df,
            x="Individualism_Score",
            y="MeanUtilitarianPreference",
            s=70,
            alpha=0.85,
            color=SCATTER_COLOR,
        )
        constant_value = analysis_df["MeanUtilitarianPreference"].iloc[0]
        ax.axhline(constant_value, color=BEST_FIT_COLOR, linestyle="--", alpha=0.7)
        title = "Individualism vs. Utilitarian Preference\n(insufficient variance for r)"
    ax.set_title(title)
    ax.set_xlabel("Individualism Score")
    ax.set_ylabel("Mean Utilitarian Preference")
    _style_axes(ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_preference_distributions(
    analysis_df: pd.DataFrame, output_path: Path, show_plot: bool = False
) -> None:
    """Plot distributions of the two preference metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    sns.histplot(
        analysis_df["MeanLegalityPreference"],
        ax=axes[0],
        color=SCATTER_COLOR,
        kde=True,
        bins=18,
    )
    axes[0].axvline(
        analysis_df["MeanLegalityPreference"].mean(),
        color=HIGHLIGHT_COLOR,
        linestyle="--",
        linewidth=2,
    )
    axes[0].set_title("Legality Preference Distribution")
    axes[0].set_xlabel("Mean Legality Preference")

    sns.histplot(
        analysis_df["MeanUtilitarianPreference"],
        ax=axes[1],
        color=SCATTER_COLOR,
        kde=True,
        bins=18,
    )
    axes[1].axvline(
        analysis_df["MeanUtilitarianPreference"].mean(),
        color=HIGHLIGHT_COLOR,
        linestyle="--",
        linewidth=2,
    )
    axes[1].set_title("Utilitarian Preference Distribution")
    axes[1].set_xlabel("Mean Utilitarian Preference")

    for ax in axes:
        _style_axes(ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_correlation_heatmap(
    analysis_df: pd.DataFrame, output_path: Path, show_plot: bool = False
) -> None:
    """Plot a correlation heatmap for the key metrics."""
    corr = analysis_df[
        [
            "Rule_of_Law_Index",
            "Individualism_Score",
            "MeanLegalityPreference",
            "MeanUtilitarianPreference",
        ]
    ].corr()
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        linewidths=0.8,
        linecolor=BACKGROUND,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap")
    _style_axes(ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_top_response_counts(
    analysis_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
    show_plot: bool = False,
) -> None:
    """Plot the countries with the largest response counts."""
    top_counts = analysis_df.nlargest(top_n, "MinObs")
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    sns.barplot(
        data=top_counts,
        x="MinObs",
        y="Country",
        color=SCATTER_COLOR,
        ax=ax,
    )
    ax.set_title(f"Top {top_n} Countries by Response Count")
    ax.set_xlabel("Minimum of Utilitarian/Legality Observations")
    ax.set_ylabel("Country")
    _style_axes(ax)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
