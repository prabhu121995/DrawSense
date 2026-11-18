"""Load and validate TOTO results from CSV / XLSX.

Required columns: draw_date, n1, n2, n3, n4, n5, n6, additional.
All other columns (draw_number, prize columns, ...) are ignored, except that
an existing draw_number is kept as a label. Invalid rows are excluded and
reported, never silently "fixed".
"""
from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import ADDITIONAL_COL, MAIN_COLS, OUTPUT_COLS, REQUIRED_COLS

log = logging.getLogger(__name__)

_COL_ALIASES = {
    "date": "draw_date", "drawdate": "draw_date",
    "drawno": "draw_number", "draw_no": "draw_number",
    "add": "additional", "additional_number": "additional", "additional_no": "additional", "bonus": "additional",
}


class DataValidationError(ValueError):
    """Raised when a file cannot be used at all (e.g. required columns missing)."""


@dataclass
class IngestResult:
    draws: pd.DataFrame
    report: dict[str, Any]
    sha256: str
    excluded: list[dict] = field(default_factory=list)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        rename[c] = _COL_ALIASES.get(key, key)
    return df.rename(columns=rename)


def read_table(content: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(io.BytesIO(content), engine="openpyxl")
    if name.endswith((".csv", ".txt")):
        return pd.read_csv(io.BytesIO(content))
    raise DataValidationError(f"Unsupported file type '{filename}'. Use .csv or .xlsx.")


_DATE_FORMATS = ["%a, %d %b %Y",       # Singapore Pools: "Mon, 20 Nov 2023"
                 "%Y-%m-%d",           # ISO (also what the artifact's history.csv uses)
                 "%Y-%m-%d %H:%M:%S",
                 "%d %b %Y",           # "20 Nov 2023"
                 "%d/%m/%Y", "%d-%m-%Y"]  # Singapore convention: day first


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse with explicit formats first. A day-first fallback must never touch ISO dates,
    otherwise 2026-11-12 would be read as 11 December."""
    s = s.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt in _DATE_FORMATS:
        todo = parsed.isna() & s.notna()
        if not todo.any():
            break
        parsed.loc[todo] = pd.to_datetime(s[todo], format=fmt, errors="coerce")
    todo = parsed.isna() & s.notna()
    if todo.any():
        parsed.loc[todo] = pd.to_datetime(s[todo], errors="coerce", dayfirst=True, format="mixed")
    return parsed


def validate_and_clean(raw: pd.DataFrame, n_max: int = 49, pick: int = 6) -> tuple[pd.DataFrame, dict, list[dict]]:
    df = _normalise_columns(raw.copy())
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}. "
                                  f"Required: {REQUIRED_COLS}. Found: {list(df.columns)}")
    has_dn = "draw_number" in df.columns
    keep = REQUIRED_COLS + (["draw_number"] if has_dn else [])
    ignored = [c for c in df.columns if c not in keep]
    df = df[keep].copy()
    n_raw = len(df)
    excluded: list[dict] = []
    original_dates = df["draw_date"].copy()

    def exclude(mask: pd.Series, reason: str) -> None:
        nonlocal df
        for idx in df.index[mask]:
            excluded.append({"row": int(idx) + 2, "draw_date": _safe(original_dates.get(idx)), "reason": reason})
        df = df[~mask]

    exclude(df[REQUIRED_COLS].isna().all(axis=1), "empty row")
    for c in MAIN_COLS + [ADDITIONAL_COL] + (["draw_number"] if has_dn else []):
        df[c] = pd.to_numeric(df[c].astype("string").str.strip(), errors="coerce")
    df["draw_date"] = _parse_dates(df["draw_date"])

    exclude(df["draw_date"].isna(), "missing/unparseable draw_date")
    exclude(df[MAIN_COLS].isna().any(axis=1), "missing main number(s)")
    exclude(df[ADDITIONAL_COL].isna(), "missing additional number")
    exclude(((df[MAIN_COLS] % 1) != 0).any(axis=1) | ((df[ADDITIONAL_COL] % 1) != 0), "non-integer number")
    exclude(((df[MAIN_COLS] < 1) | (df[MAIN_COLS] > n_max)).any(axis=1), f"main number outside 1-{n_max}")
    exclude((df[ADDITIONAL_COL] < 1) | (df[ADDITIONAL_COL] > n_max), f"additional number outside 1-{n_max}")
    exclude(df[MAIN_COLS].nunique(axis=1) != pick, "main numbers not unique")
    exclude(df[MAIN_COLS].eq(df[ADDITIONAL_COL], axis=0).any(axis=1), "additional number repeats a main number")

    for c in MAIN_COLS + [ADDITIONAL_COL]:
        df[c] = df[c].astype(int)
    df[MAIN_COLS] = np.sort(df[MAIN_COLS].to_numpy(), axis=1)

    # Duplicates: identical draw on the same date -> keep first; same date with different numbers -> exclude all.
    exact_dup = df.duplicated(subset=REQUIRED_COLS, keep="first")
    n_exact = int(exact_dup.sum())
    exclude(exact_dup, "duplicate draw (identical) - kept first")
    conflict = df.duplicated(subset=["draw_date"], keep=False)
    if has_dn:  # two different draws on one date are legitimate if their draw numbers differ
        conflict &= df.duplicated(subset=["draw_date", "draw_number"], keep=False)
    n_conflict = int(conflict.sum())
    exclude(conflict, "conflicting rows for the same draw (different numbers) - all excluded")

    sort_cols = ["draw_date", "draw_number"] if has_dn and df["draw_number"].notna().all() else ["draw_date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    if not has_dn or df["draw_number"].isna().any():
        df["draw_number"] = np.arange(1, len(df) + 1)  # generated label, chronological
    df["draw_number"] = df["draw_number"].astype(int)
    df = df[OUTPUT_COLS]

    warnings: list[str] = []
    if ignored:
        warnings.append(f"Ignored columns (not used): {ignored}")
    if not has_dn:
        warnings.append("No draw_number column: draws are labelled 1..N in date order.")
    if len(df) < 150:
        warnings.append(f"Only {len(df)} valid draws: model comparisons will have very low statistical power.")
    report = {
        "n_rows_raw": n_raw, "n_valid_draws": int(len(df)), "n_excluded": len(excluded),
        "n_exact_duplicates": n_exact, "n_conflicting_duplicates": n_conflict,
        "date_min": df["draw_date"].min().strftime("%Y-%m-%d") if len(df) else None,
        "date_max": df["draw_date"].max().strftime("%Y-%m-%d") if len(df) else None,
        "warnings": warnings, "excluded_rows": excluded[:200],
    }
    if len(df) == 0:
        raise DataValidationError("No valid draws after validation. " + "; ".join(e["reason"] for e in excluded[:5]))
    return df, report, excluded


def _safe(v):
    try:
        return None if pd.isna(v) else str(v)
    except (TypeError, ValueError):
        return str(v)


def ingest_bytes(content: bytes, filename: str) -> IngestResult:
    draws, report, excluded = validate_and_clean(read_table(content, filename))
    report["filename"] = filename
    log.info("Ingested %s: %d valid draws, %d excluded", filename, len(draws), len(excluded))
    return IngestResult(draws=draws, report=report, sha256=hashlib.sha256(content).hexdigest(), excluded=excluded)


def ingest_file(path: str | Path) -> IngestResult:
    path = Path(path)
    return ingest_bytes(path.read_bytes(), path.name)


def draws_to_matrix(draws: pd.DataFrame, n_max: int = 49) -> np.ndarray:
    """(T, n_max) 0/1 indicator matrix of main numbers, rows in chronological order."""
    T = len(draws)
    M = np.zeros((T, n_max), dtype=np.int8)
    M[np.arange(T)[:, None], draws[MAIN_COLS].to_numpy() - 1] = 1
    return M


def additional_to_matrix(draws: pd.DataFrame, n_max: int = 49) -> np.ndarray:
    T = len(draws)
    A = np.zeros((T, n_max), dtype=np.int8)
    A[np.arange(T), draws[ADDITIONAL_COL].to_numpy() - 1] = 1
    return A
