"""Aggregation and statistical helpers for the Moral Machine analysis."""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress, pearsonr, spearmanr

from .data_io import PYARROW_AVAILABLE, clean_country_series, detect_country_column

DEBUG_LOG_PATH = Path(
    r"c:\Users\AnthonyPC\Documents\GitHub\COG260-Project-Moral-Machine-Analysis\.cursor\debug.log"
)
DEBUG_SESSION_ID = "debug-session"
DEBUG_RUN_ID = "pre-fix"


def _debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": DEBUG_RUN_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

REQUIRED_MORAL_COLS = [
    "PedPed",
    "CrossingSignal",
    "Saved",
    "DiffNumberOFCharacters",
    "AttributeLevel",
    "Dog",
    "Cat",
]


def aggregate_moral_preferences(
    path: str | Path, chunk_rows: int = 500_000
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Stream the Moral Machine CSV and derive per-country utilitarian/legality means.

    Utilitarian: include all dilemmas but drop any with pets or tied counts, and only
    score the 'More' side (Saved=+1, otherwise -1). Legality: keep pedestrian vs.
    pedestrian dilemmas with traffic lights and score lawful choices as +1, unlawful
    as -1.
    """
    # region agent log
    _debug_log(
        "H1",
        "metrics.py:aggregate_moral_preferences:entry",
        "aggregate_moral_preferences called",
        {
            "chunk_rows": chunk_rows,
            "pyarrow_available": PYARROW_AVAILABLE,
            "pandas_version": pd.__version__,
        },
    )
    # endregion agent log

    probe = pd.read_csv(path, nrows=0)
    columns = probe.columns.tolist()
    country_col = detect_country_column(columns)
    missing = sorted(set(REQUIRED_MORAL_COLS) - set(columns))
    if missing:
        raise KeyError(f"Required columns missing: {missing}")

    usecols = sorted(set(REQUIRED_MORAL_COLS) | {country_col})
    read_kwargs: dict[str, Any] = {"usecols": usecols, "chunksize": chunk_rows}
    if PYARROW_AVAILABLE:
        read_kwargs.update({"engine": "pyarrow", "dtype_backend": "pyarrow"})

    # region agent log
    _debug_log(
        "H1",
        "metrics.py:aggregate_moral_preferences:read_kwargs",
        "read_csv kwargs prepared",
        {
            "engine": read_kwargs.get("engine"),
            "chunksize": read_kwargs.get("chunksize"),
            "dtype_backend": read_kwargs.get("dtype_backend"),
            "usecols_count": len(usecols),
        },
    )
    # endregion agent log

    read_attempt = 1
    try:
        reader = pd.read_csv(path, **read_kwargs)
    except TypeError as exc:
        # region agent log
        _debug_log(
            "H2",
            "metrics.py:aggregate_moral_preferences:read_csv_type_error",
            "read_csv raised TypeError; retrying without dtype_backend",
            {
                "attempt": read_attempt,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "engine": read_kwargs.get("engine"),
            },
        )
        # endregion agent log
        read_kwargs.pop("dtype_backend", None)
        read_attempt = 2
        try:
            reader = pd.read_csv(path, **read_kwargs)
        except Exception as exc2:
            # region agent log
            _debug_log(
                "H1",
                "metrics.py:aggregate_moral_preferences:read_csv_exception",
                "read_csv failed after retry",
                {
                    "attempt": read_attempt,
                    "error_type": type(exc2).__name__,
                    "error": str(exc2),
                    "engine": read_kwargs.get("engine"),
                    "chunksize": read_kwargs.get("chunksize"),
                },
            )
            # endregion agent log
            raise
    except ValueError as exc:
        if (
            read_kwargs.get("engine") == "pyarrow"
            and "chunksize" in str(exc).lower()
        ):
            # region agent log
            _debug_log(
                "H1",
                "metrics.py:aggregate_moral_preferences:read_csv_pyarrow_chunksize",
                "pyarrow does not support chunksize; retrying with default engine",
                {
                    "attempt": read_attempt,
                    "error": str(exc),
                    "engine": read_kwargs.get("engine"),
                    "chunksize": read_kwargs.get("chunksize"),
                },
            )
            # endregion agent log
            read_kwargs.pop("engine", None)
            read_kwargs.pop("dtype_backend", None)
            read_attempt = 2
            try:
                reader = pd.read_csv(path, **read_kwargs)
            except Exception as exc2:
                # region agent log
                _debug_log(
                    "H1",
                    "metrics.py:aggregate_moral_preferences:read_csv_exception",
                    "read_csv failed after pyarrow fallback",
                    {
                        "attempt": read_attempt,
                        "error_type": type(exc2).__name__,
                        "error": str(exc2),
                        "engine": read_kwargs.get("engine"),
                        "chunksize": read_kwargs.get("chunksize"),
                    },
                )
                # endregion agent log
                raise
        else:
            # region agent log
            _debug_log(
                "H1",
                "metrics.py:aggregate_moral_preferences:read_csv_exception",
                "read_csv failed",
                {
                    "attempt": read_attempt,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "engine": read_kwargs.get("engine"),
                    "chunksize": read_kwargs.get("chunksize"),
                },
            )
            # endregion agent log
            raise
    except Exception as exc:
        # region agent log
        _debug_log(
            "H1",
            "metrics.py:aggregate_moral_preferences:read_csv_exception",
            "read_csv failed",
            {
                "attempt": read_attempt,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "engine": read_kwargs.get("engine"),
                "chunksize": read_kwargs.get("chunksize"),
            },
        )
        # endregion agent log
        raise

    # region agent log
    _debug_log(
        "H3",
        "metrics.py:aggregate_moral_preferences:read_csv_success",
        "read_csv returned reader",
        {
            "attempt": read_attempt,
            "reader_type": type(reader).__name__,
            "engine": read_kwargs.get("engine"),
        },
    )
    # endregion agent log

    util_summaries: list[pd.DataFrame] = []
    legal_summaries: list[pd.DataFrame] = []
    total_rows_scanned = 0
    countries_seen: set[str] = set()

    for chunk in reader:
        total_rows_scanned += len(chunk)
        chunk = chunk.rename(columns={country_col: "Country"})
        chunk["Country"] = clean_country_series(chunk["Country"])
        countries_seen.update(chunk["Country"].dropna().unique().tolist())

        for column in ("PedPed", "CrossingSignal", "Saved", "DiffNumberOFCharacters"):
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        for column in ("Dog", "Cat"):
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").fillna(0)

        attr_level = chunk["AttributeLevel"].fillna("").str.lower()
        animals_removed = (chunk["Dog"] == 0) & (chunk["Cat"] == 0)
        diff_nonzero = chunk["DiffNumberOFCharacters"].fillna(0) != 0
        util_mask = animals_removed & diff_nonzero & (attr_level == "more")

        if util_mask.any():
            util_chunk = chunk.loc[util_mask, ["Country", "Saved"]].copy()
            saved_more = util_chunk["Saved"].fillna(0).astype(int)
            util_chunk["utilitarian_score"] = np.where(saved_more == 1, 1, -1)
            util_summary = util_chunk.groupby("Country").agg(
                utilitarian_sum=("utilitarian_score", "sum"),
                utilitarian_obs=("utilitarian_score", "size"),
            )
            util_summaries.append(util_summary)

        pedestrian = chunk[chunk["PedPed"] == 1]
        if pedestrian.empty:
            continue

        traffic_lights = pedestrian[pedestrian["CrossingSignal"] > 0]
        if traffic_lights.empty:
            continue

        saved_light = traffic_lights["Saved"].fillna(0).astype(int)
        signal = traffic_lights["CrossingSignal"]
        legal_mask = signal.isin([1, 2])
        legal_choice = ((signal == 1) & (saved_light == 1)) | (
            (signal == 2) & (saved_light == 0)
        )
        legality_score = pd.Series(
            np.where(legal_mask, np.where(legal_choice, 1, -1), np.nan),
            index=traffic_lights.index,
        )

        legal_summary = traffic_lights.assign(legality_score=legality_score).groupby(
            "Country"
        ).agg(
            legality_sum=("legality_score", lambda s: np.nansum(s)),
            legality_obs=("legality_score", lambda s: s.notna().sum()),
        )
        legal_summaries.append(legal_summary)

    if not util_summaries:
        raise ValueError(
            "No utilitarian dilemmas met the filters (no animals, non-tied counts, AttributeLevel='More')."
        )
    if not legal_summaries:
        raise ValueError(
            "No traffic-light dilemmas were found to compute legality preferences."
        )

    util_aggregated = pd.concat(util_summaries).groupby("Country").sum()
    util_aggregated = util_aggregated[util_aggregated["utilitarian_obs"] > 0]
    util_aggregated["Mean_Utilitarian"] = (
        util_aggregated["utilitarian_sum"] / util_aggregated["utilitarian_obs"]
    )

    legal_aggregated = pd.concat(legal_summaries).groupby("Country").sum()
    legal_aggregated = legal_aggregated[legal_aggregated["legality_obs"] > 0]
    legal_aggregated["Mean_Legality"] = (
        legal_aggregated["legality_sum"] / legal_aggregated["legality_obs"]
    )

    result = (
        util_aggregated[["Mean_Utilitarian", "utilitarian_obs"]]
        .join(legal_aggregated[["Mean_Legality", "legality_obs"]], how="inner")
        .sort_index()
    )
    result.index.name = "Country"

    return result, {
        "rows_scanned": total_rows_scanned,
        "countries_seen": len(countries_seen),
    }


def compute_correlations(analysis_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute Pearson and Spearman correlations for the two hypotheses."""
    results: dict[str, dict[str, float]] = {}
    pairs = {
        "rule_of_law_vs_legality": ("Rule_of_Law_Index", "MeanLegalityPreference"),
        "individualism_vs_utilitarian": (
            "Individualism_Score",
            "MeanUtilitarianPreference",
        ),
    }

    for key, (x_col, y_col) in pairs.items():
        if analysis_df[x_col].nunique() < 2 or analysis_df[y_col].nunique() < 2:
            results[key] = {
                "pearson_r": float("nan"),
                "pearson_p": float("nan"),
                "spearman_r": float("nan"),
                "spearman_p": float("nan"),
            }
            continue
        pearson_r, pearson_p = pearsonr(analysis_df[x_col], analysis_df[y_col])
        spearman_r, spearman_p = spearmanr(analysis_df[x_col], analysis_df[y_col])
        results[key] = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        }

    return results


def fit_linear_regression(
    analysis_df: pd.DataFrame, x_col: str, y_col: str
) -> dict[str, float] | None:
    """Fit a simple linear regression if there is sufficient variance."""
    if analysis_df[x_col].nunique() < 2 or analysis_df[y_col].nunique() < 2:
        return None

    result = linregress(analysis_df[x_col], analysis_df[y_col])
    return {
        "slope": float(result.slope),
        "intercept": float(result.intercept),
        "rvalue": float(result.rvalue),
        "pvalue": float(result.pvalue),
        "stderr": float(result.stderr),
        "n": int(len(analysis_df)),
    }


def find_outlier_labels(
    df: pd.DataFrame, x_col: str, y_col: str, top_n: int = 8
) -> pd.DataFrame:
    """Label the points with the largest absolute residuals from the best-fit line."""
    slope, intercept = np.polyfit(df[x_col], df[y_col], 1)
    return (
        df.assign(_residual=(df[y_col] - (slope * df[x_col] + intercept)).abs())
        .nlargest(top_n, "_residual")
        .drop(columns="_residual")
    )
