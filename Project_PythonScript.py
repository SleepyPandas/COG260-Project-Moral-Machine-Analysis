"""
NOTE : This script is designed to be run as a standalone Python file.

THE CSV IS VERY LARGE so the runtime will be > 1 minute 


Further you will require 3 CSV files in a folder named Data in the same directory as this script:
- SharedResponses.csv
- RuleOfLaw.csv
- IndividualisticRanking.csv

these can be obtained from the Moral Machine Study data release:
https://osf.io/3hvt2/overview?view_only=4bb49492edee4a8eb1758552a362a2cf

Datasets/Moral Machine Data/SharedResponses.csv.tar.gz

And in the Github Repository


"""


from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycountry
import seaborn as sns
from scipy.stats import pearsonr

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
sns.set_theme(style="whitegrid", context="talk")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "Data"  # folder containing CSV inputs
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"  # where plots are saved

# Consistent colors for plots (tuned for the dark background set in main()).
# Cyan points with a red best-fit line for both plots.
SCATTER_COLOR = "#2ca9c0"
BEST_FIT_COLOR = "#c23b22"
ANNOTATION_FACE_COLOR = "#111111"

# Optional speed-up if pyarrow is installed; we fall back silently if not.
PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
COUNTRY_CANDIDATES = [
    "Country",
    "country",
    "RespondentCountry",
    "Respondent_Country",
    "Nationality",
    "UserCountry3",
]

RULE_OF_LAW_ALIASES = {
    "Rule_of_Law_Index": [
        "Rule_of_Law_Index",
        "Rule_of_Law",
        "RuleOfLaw",
        "rol_index",
        "WJP Rule of Law Index: Overall Score",
    ],
}
INDIVIDUALISM_ALIASES = {
    "Individualism_Score": [
        "Individualism_Score",
        "Individualism",
        "individualism",
        "IndividualisticRanking",
        "Individualistic",
    ]
}

ISO_OVERRIDES = {"XKX": "Kosovo"}
ISO3_TO_NAME = {c.alpha_3: c.name for c in pycountry.countries}
ISO3_TO_NAME.update(ISO_OVERRIDES)


def detect_country_column(columns):
    """Return the first column name that looks like a country indicator."""
    for candidate in COUNTRY_CANDIDATES:
        if candidate in columns:
            return candidate
    for column in columns:
        if "country" in column.lower():
            return column
    raise KeyError("No country-like column found")


def normalize_country_values(series):
    """Coerce ISO3 codes to full country names; leave other values untouched."""

    def convert(value):
        if isinstance(value, str):
            code = value.strip()
            if len(code) == 3 and code.isalpha():
                upper = code.upper()
                return ISO3_TO_NAME.get(upper, upper)
        return value

    return series.apply(convert)


def standardize_country_column(df):
    """Rename the detected country column to `Country` and normalise its values."""
    country_col = detect_country_column(df.columns)
    if country_col != "Country":
        df = df.rename(columns={country_col: "Country"})
    df["Country"] = normalize_country_values(df["Country"])
    return df


def rename_with_aliases(df, alias_map):
    """Map heterogeneous column labels into a predictable schema."""
    renamed = {}
    for target, candidates in alias_map.items():
        for column in candidates:
            if column in df.columns:
                renamed[column] = target
                break
        if target not in renamed.values() and target not in df.columns:
            raise KeyError(f"Column providing '{target}' not found")
    return df.rename(columns=renamed)


def maybe_transpose_country_rows(df):
    """Handle lookup tables that encode countries as columns rather than rows."""
    if "Country" in df.columns and len(df) < len(df.columns):
        transposed = df.set_index("Country").T
        return transposed.reset_index().rename(columns={"index": "Country"})
    return df


def load_lookup_table(path, alias_map):
    """Load, tidy, and narrow a reference CSV down to Country plus target indicators."""
    frame = pd.read_csv(path)
    frame = maybe_transpose_country_rows(frame)
    frame = standardize_country_column(frame)
    frame = rename_with_aliases(frame, alias_map)
    keep_columns = ["Country", *alias_map.keys()]
    return frame.loc[:, keep_columns]


# Change Depending on Resources
def aggregate_moral_preferences(path, chunk_rows=1_000_000):
    """
    Stream the Moral Machine CSV and derive per-country utilitarian/legality means.

    Utilitarian: include all dilemmas but drop any with pets or tied counts, and only
    score the 'More' side (Saved=+1, otherwise -1). Legality: keep pedestrian vs.
    pedestrian dilemmas with traffic lights and score lawful choices as +1, unlawful
    as -1.
    """
    # Inspect the header to find required columns without loading the entire file.
    probe = pd.read_csv(path, nrows=0)
    columns = probe.columns.tolist()
    country_col = detect_country_column(columns)
    required_cols = {
        "PedPed",
        "CrossingSignal",
        "Saved",
        "DiffNumberOFCharacters",
        "AttributeLevel",
        "Dog",
        "Cat",
    }
    missing = sorted(required_cols - set(columns))
    if missing:
        raise KeyError(f"Required columns missing: {missing}")

    usecols = sorted(required_cols | {country_col})
    read_kwargs = {"usecols": usecols, "chunksize": chunk_rows}
    if PYARROW_AVAILABLE:
        read_kwargs.update({"engine": "pyarrow", "dtype_backend": "pyarrow"})

    try:
        reader = pd.read_csv(path, **read_kwargs)
    except TypeError:
        read_kwargs.pop("dtype_backend", None)
        reader = pd.read_csv(path, **read_kwargs)

    util_summaries = []  # holds per-chunk utilitarian aggregates
    legal_summaries = []  # holds per-chunk legality aggregates
    total_rows_scanned = 0

    for chunk in reader:
        total_rows_scanned += len(chunk)
        chunk = chunk.rename(columns={country_col: "Country"})
        chunk["Country"] = normalize_country_values(chunk["Country"])

        # Utilitarian preference: include broader dilemmas but exclude animals and ties.
        attr_level = chunk["AttributeLevel"].fillna("").str.lower()
        animals_removed = (chunk["Dog"].fillna(0) == 0) & (chunk["Cat"].fillna(0) == 0)
        diff_nonzero = chunk["DiffNumberOFCharacters"].fillna(0) != 0
        utilitarian_more = chunk[
            animals_removed & diff_nonzero & (attr_level == "more")
        ]
        if not utilitarian_more.empty:
            saved_more = utilitarian_more["Saved"].fillna(0).astype(int)
            utilitarian_scores = np.where(saved_more == 1, 1, -1)
            util_summary = (
                utilitarian_more.assign(utilitarian_score=utilitarian_scores)
                .groupby("Country")
                .agg(
                    utilitarian_sum=("utilitarian_score", "sum"),
                    utilitarian_obs=("utilitarian_score", "size"),
                )
            )
            util_summaries.append(util_summary)

        # Legality preference: pedestrian vs pedestrian dilemmas with signals.
        pedestrian = chunk[chunk["PedPed"] == 1]
        if pedestrian.empty:
            continue

        traffic_lights = pedestrian[pedestrian["CrossingSignal"] > 0]
        if traffic_lights.empty:
            continue

        saved_light = traffic_lights["Saved"].fillna(0).astype(int)
        legality_score = pd.Series(np.nan, index=traffic_lights.index)
        legal_mask = traffic_lights["CrossingSignal"].isin([1, 2])
        legality_score.loc[legal_mask] = np.where(
            (
                (traffic_lights.loc[legal_mask, "CrossingSignal"] == 1)
                & (saved_light.loc[legal_mask] == 1)
            )
            | (
                (traffic_lights.loc[legal_mask, "CrossingSignal"] == 2)
                & (saved_light.loc[legal_mask] == 0)
            ),
            1,
            -1,
        )

        legal_summary = (
            traffic_lights.assign(legality_score=legality_score)
            .groupby("Country")
            .agg(
                legality_sum=("legality_score", lambda s: np.nansum(s)),
                legality_obs=("legality_score", lambda s: s.notna().sum()),
            )
        )
        legal_summaries.append(legal_summary)

    if not util_summaries:
        raise ValueError(
            "No utilitarian dilemmas met the filters (no animals, non-tied counts, AttributeLevel='More')."
        )

    util_aggregated = pd.concat(util_summaries).groupby("Country").sum()
    util_aggregated = util_aggregated[util_aggregated["utilitarian_obs"] > 0]
    util_aggregated["Mean_Utilitarian"] = (
        util_aggregated["utilitarian_sum"] / util_aggregated["utilitarian_obs"]
    )

    if not legal_summaries:
        raise ValueError(
            "No traffic-light dilemmas were found to compute legality preferences."
        )

    legal_aggregated = pd.concat(legal_summaries).groupby("Country").sum()
    legal_aggregated = legal_aggregated[legal_aggregated["legality_obs"] > 0]
    legal_aggregated["Mean_Legality"] = (
        legal_aggregated["legality_sum"] / legal_aggregated["legality_obs"]
    )

    result = (
        util_aggregated[["Mean_Utilitarian"]]
        .join(legal_aggregated[["Mean_Legality"]], how="inner")
        .sort_index()
    )
    result.index.name = "Country"
    print(
        f"Processed {total_rows_scanned:,} rows without loading the full dataset at once."
    )
    return result


def compute_correlations(analysis_df):
    """Compute Pearson correlations for the two hypotheses."""
    if len(analysis_df) < 3:
        raise ValueError("Need at least 3 countries to compute robust correlations.")

    rol_r, rol_p = pearsonr(
        analysis_df["Rule_of_Law_Index"],
        analysis_df["MeanLegalityPreference"],
    )

    if analysis_df["MeanUtilitarianPreference"].nunique() > 1:
        indiv_r, indiv_p = pearsonr(
            analysis_df["Individualism_Score"],
            analysis_df["MeanUtilitarianPreference"],
        )
    else:
        indiv_r, indiv_p = (np.nan, np.nan)

    return (rol_r, rol_p), (indiv_r, indiv_p)


def find_outlier_labels(df, x_col, y_col, top_n=8):
    """Label the points with the largest absolute residuals from the best-fit line."""
    x_vals = df[x_col]
    y_vals = df[y_col]
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    return (
        df.assign(_residual=(y_vals - (slope * x_vals + intercept)).abs())
        .nlargest(top_n, "_residual")
        .drop(columns="_residual")
    )


def plot_legality_vs_rule_of_law(
    analysis_df, rol_r, rol_p, outlier_labels, output_path, show_plot=False
):
    """Create and save the legality vs. rule-of-law scatter with regression line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(
        ax=ax,
        data=analysis_df,
        x="Rule_of_Law_Index",
        y="MeanLegalityPreference",
        scatter_kws={"s": 70, "alpha": 0.85, "color": SCATTER_COLOR},
        line_kws={"color": BEST_FIT_COLOR, "linewidth": 2.5},
        color=SCATTER_COLOR,
    )
    if outlier_labels is not None and len(outlier_labels):
        for _, row in outlier_labels.iterrows():
            ax.annotate(
                row["Country"],
                (row["Rule_of_Law_Index"], row["MeanLegalityPreference"]),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=10,
                color="#f7b6d2",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": ANNOTATION_FACE_COLOR,
                    "alpha": 0.7,
                    "edgecolor": BEST_FIT_COLOR,
                },
            )
    ax.set_title(
        f"Rule of Law vs. Legality Preference\n(r = {rol_r:.2f}, p = {rol_p:.3f})"
    )
    ax.set_xlabel("Rule of Law Index")
    ax.set_ylabel("Mean Legality Preference")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_utilitarian_vs_individualism(
    analysis_df, indiv_r, indiv_p, outlier_labels, output_path, show_plot=False
):
    """Create and save the utilitarian vs. individualism scatter with regression line."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if analysis_df["MeanUtilitarianPreference"].nunique() > 1:
        sns.regplot(
            ax=ax,
            data=analysis_df,
            x="Individualism_Score",
            y="MeanUtilitarianPreference",
            scatter_kws={"s": 70, "alpha": 0.85, "color": SCATTER_COLOR},
            line_kws={"color": BEST_FIT_COLOR, "linewidth": 2.5},
        )
        if outlier_labels is not None and len(outlier_labels):
            for _, row in outlier_labels.iterrows():
                ax.annotate(
                    row["Country"],
                    (row["Individualism_Score"], row["MeanUtilitarianPreference"]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=10,
                    color="#fff5f0",
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": ANNOTATION_FACE_COLOR,
                        "alpha": 0.7,
                        "edgecolor": BEST_FIT_COLOR,
                    },
                )
        ax.set_title(
            f"Individualism vs. Utilitarian Preference\n(r = {indiv_r:.2f}, p = {indiv_p:.3f})"
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
        ax.set_title(
            "Individualism vs. Utilitarian Preference\n(insufficient variance for r)"
        )
    ax.set_xlabel("Individualism Score")
    ax.set_ylabel("Mean Utilitarian Preference")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    """CLI for locating data, controlling chunk size, and toggling plot display."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-country Moral Machine preferences and output two scatter plots "
            "linking legality to rule-of-law and utilitarianism to individualism."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing SharedResponses.csv, RuleOfLaw.csv, and IndividualisticRanking.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the generated plot images",
    )
    # For this, it is set to 8_000_000 however for this data set the chunks will never consume more than 4GB of RAM
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=8_000_000,
        help="Number of rows to process per chunk from the Moral Machine CSV",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots after saving them",
    )
    return parser.parse_args()


def main():
    # Parse CLI arguments and resolve paths.
    args = parse_args()
    data_dir: Path = args.data_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)  # ensure output path exists

    # Required input files.
    moral_machine_path = data_dir / "SharedResponses.csv"
    rule_of_law_path = data_dir / "RuleOfLaw.csv"
    individualism_path = data_dir / "IndividualisticRanking.csv"

    print(f"Using data directory: {data_dir}")
    for path in (moral_machine_path, rule_of_law_path, individualism_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        print(f" - {path.name}: found ({path.stat().st_size / 1_048_576:.1f} MB)")

    # Peek at a handful of rows to confirm the schema.
    probe_sample = pd.read_csv(
        moral_machine_path,
        usecols=[
            "UserCountry3",
            "PedPed",
            "CrossingSignal",
            "Saved",
            "DefaultChoice",
            "NonDefaultChoice",
            "NumberOfCharacters",
            "DiffNumberOFCharacters",
        ],
        nrows=5,
    )
    print("\nSample of Moral Machine rows:")
    print(probe_sample.to_string(index=False))

    # Load cultural indicators and standardize column names.
    rule_of_law_df = load_lookup_table(rule_of_law_path, RULE_OF_LAW_ALIASES)
    individualism_df = load_lookup_table(individualism_path, INDIVIDUALISM_ALIASES)
    print(f"\nRule-of-law rows: {len(rule_of_law_df)}")
    print(rule_of_law_df.head().to_string(index=False))
    print(f"\nIndividualism rows: {len(individualism_df)}")
    print(individualism_df.head().to_string(index=False))

    # Stream the large Moral Machine file in chunks to derive per-country means.
    moral_country_means = aggregate_moral_preferences(
        moral_machine_path, chunk_rows=args.chunk_rows
    )
    print(f"\nCountries with qualifying dilemmas: {len(moral_country_means)}")
    print(moral_country_means.head().to_string())

    # Merge preferences with cultural indicators and drop incomplete rows.
    merged_preferences = (
        moral_country_means.reset_index()
        .merge(rule_of_law_df, on="Country", how="inner")
        .merge(individualism_df, on="Country", how="inner")
        .dropna()
    )
    analysis_df = merged_preferences.rename(
        columns={
            "Mean_Utilitarian": "MeanUtilitarianPreference",
            "Mean_Legality": "MeanLegalityPreference",
        }
    )

    print(f"\nMerged dataset rows: {len(analysis_df)}")
    print(analysis_df.head().to_string(index=False))

    # Hypothesis tests: Pearson correlations for H1 and H2.
    (rol_r, rol_p), (indiv_r, indiv_p) = compute_correlations(analysis_df)
    print(f"\nRule of Law vs. Legality Preference: r = {rol_r:.3f}, p = {rol_p:.4f}")
    if not np.isnan(indiv_r):
        print(
            f"Individualism vs. Utilitarian Preference: r = {indiv_r:.3f}, p = {indiv_p:.4f}"
        )
    else:
        print("Individualism vs. Utilitarian Preference: insufficient variance for r")

    plt.style.use("dark_background")
    # Label the largest residuals to contextualize the plots.
    legality_outliers = find_outlier_labels(
        analysis_df, "Rule_of_Law_Index", "MeanLegalityPreference", top_n=8
    )
    util_outliers = None
    if analysis_df["MeanUtilitarianPreference"].nunique() > 1:
        util_outliers = find_outlier_labels(
            analysis_df, "Individualism_Score", "MeanUtilitarianPreference", top_n=8
        )

    rol_plot_path = output_dir / "rule_of_law_vs_legality.png"
    util_plot_path = output_dir / "individualism_vs_utilitarian.png"

    plot_legality_vs_rule_of_law(
        analysis_df,
        rol_r,
        rol_p,
        legality_outliers,
        rol_plot_path,
        show_plot=args.show,
    )
    plot_utilitarian_vs_individualism(
        analysis_df,
        indiv_r,
        indiv_p,
        util_outliers,
        util_plot_path,
        show_plot=args.show,
    )

    print(f"\nSaved plots to:")
    print(f" - {rol_plot_path}")
    print(f" - {util_plot_path}")


if __name__ == "__main__":
    main()
