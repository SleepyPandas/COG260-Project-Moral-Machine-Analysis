"""Data loading, normalization, and schema handling utilities."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Iterable

import pandas as pd
import pycountry

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
ISO3_TO_NAME = {country.alpha_3: country.name for country in pycountry.countries}
ISO3_TO_NAME.update(ISO_OVERRIDES)

COUNTRY_NAME_ALIASES = {
    "Bahamas": "The Bahamas",
    "Gambia": "The Gambia",
    "South Korea": "Korea, Rep.",
    "Korea, Republic of": "Korea, Rep.",
    "Russia": "Russian Federation",
    "Czech Republic": "Czechia",
    "Slovakia": "Slovak Republic",
    "Iran": "Iran, Islamic Rep.",
    "Iran, Islamic Republic of": "Iran, Islamic Rep.",
    "Egypt": "Egypt, Arab Rep.",
    "Venezuela": "Venezuela, RB",
    "Venezuela, Bolivarian Republic of": "Venezuela, RB",
    "Hong Kong": "Hong Kong SAR, China",
    "Hong Kong SAR": "Hong Kong SAR, China",
    "Turkey": "Türkiye",
    "Viet Nam": "Vietnam",
    "Bolivia, Plurinational State of": "Bolivia",
    "United States of America": "United States",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Republic of Moldova": "Moldova",
    "Tanzania, United Republic of": "Tanzania",
    "Congo, The Democratic Republic of the": "Congo, Dem. Rep.",
    "Congo, Republic of the": "Congo, Rep.",
    "Cote d'Ivoire": "Côte d'Ivoire",
}


def detect_country_column(columns: Iterable[str]) -> str:
    """Return the first column name that looks like a country indicator."""
    for candidate in COUNTRY_CANDIDATES:
        if candidate in columns:
            return candidate
    for column in columns:
        if "country" in column.lower():
            return column
    raise KeyError("No country-like column found")


def normalize_country_values(series: pd.Series) -> pd.Series:
    """Coerce ISO3 codes to full country names; leave other values untouched."""

    def convert(value):
        if isinstance(value, str):
            code = value.strip()
            if len(code) == 3 and code.isalpha():
                upper = code.upper()
                return ISO3_TO_NAME.get(upper, upper)
        return value

    return series.apply(convert)


def apply_country_aliases(series: pd.Series) -> pd.Series:
    """Normalize common country naming variants to a shared label."""
    return series.replace(COUNTRY_NAME_ALIASES)


def clean_country_series(series: pd.Series) -> pd.Series:
    """Normalize country values and resolve common aliases."""
    normalized = normalize_country_values(series)
    normalized = normalized.apply(
        lambda value: value.strip() if isinstance(value, str) else value
    )
    return apply_country_aliases(normalized)


def standardize_country_column(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the detected country column to `Country` and normalise its values."""
    country_col = detect_country_column(df.columns)
    if country_col != "Country":
        df = df.rename(columns={country_col: "Country"})
    df["Country"] = clean_country_series(df["Country"])
    return df


def rename_with_aliases(df: pd.DataFrame, alias_map: dict[str, list[str]]) -> pd.DataFrame:
    """Map heterogeneous column labels into a predictable schema."""
    renamed: dict[str, str] = {}
    for target, candidates in alias_map.items():
        for column in candidates:
            if column in df.columns:
                renamed[column] = target
                break
        if target not in renamed.values() and target not in df.columns:
            raise KeyError(f"Column providing '{target}' not found")
    return df.rename(columns=renamed)


def maybe_transpose_country_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Handle lookup tables that encode countries as columns rather than rows."""
    if "Country" in df.columns and len(df) < len(df.columns):
        transposed = df.set_index("Country").T
        return transposed.reset_index().rename(columns={"index": "Country"})
    return df


def load_lookup_table(path: Path, alias_map: dict[str, list[str]]) -> pd.DataFrame:
    """Load, tidy, and narrow a reference CSV down to Country plus target indicators."""
    frame = pd.read_csv(path)
    frame = maybe_transpose_country_rows(frame)
    frame = standardize_country_column(frame)
    frame = rename_with_aliases(frame, alias_map)
    keep_columns = ["Country", *alias_map.keys()]
    frame = frame.loc[:, keep_columns]
    for column in alias_map.keys():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["Country"])
