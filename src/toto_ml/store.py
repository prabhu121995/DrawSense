"""The results store is your raw results file, ``toto_results.csv``.

* Reading: only draw_date, n1..n6, additional are used (validated).
* Adding: new draws are appended in the file's own date format; every other
  column (draw_number, prize columns, ...) is left blank for new rows and is
  never changed for existing rows.
* Before the first change a one-time backup ``toto_results.backup.csv`` is written.

notebook.ipynb trains from this file and DrawSense predicts from it.
"""
from __future__ import annotations

import os
import shutil
from datetime import date
from pathlib import Path

import pandas as pd

from .config import N_NUMBERS, PICK, PROJECT_ROOT
from .data import ingest_file, validate_and_clean
from .data.ingest import _parse_dates, read_table
from .data.schema import MAIN_COLS, REQUIRED_COLS

STORE_PATH = Path(os.environ.get("TOTO_ML_STORE", PROJECT_ROOT / "toto_results.csv"))
SG_DATE_FORMAT = "%a, %d %b %Y"  # "Mon, 20 Nov 2023", as scraped from Singapore Pools


def _p(path: Path | None) -> Path:
    """Resolve at call time so STORE_PATH can be changed (tests, other locations)."""
    return Path(path) if path else STORE_PATH


def backup_path(path: Path | None = None) -> Path:
    path = _p(path)
    return path.with_name(path.stem + ".backup" + path.suffix)


def _read_raw(path: Path) -> pd.DataFrame:
    # strings in, strings out: existing rows are written back exactly as they were
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _write_raw(raw: pd.DataFrame, path: Path) -> None:
    bak = backup_path(path)
    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)  # one-time safety copy of the original file
    tmp = path.with_suffix(".tmp")
    raw.to_csv(tmp, index=False)
    tmp.replace(path)  # atomic: never leaves a half-written file


def _date_format(raw: pd.DataFrame) -> str:
    """Write new dates in the same style as the file already uses."""
    sample = next((v for v in raw.get("draw_date", []) if str(v).strip()), "")
    try:
        pd.to_datetime(sample, format=SG_DATE_FORMAT)
        return SG_DATE_FORMAT
    except (ValueError, TypeError):
        return "%Y-%m-%d"


def _sort_raw(raw: pd.DataFrame) -> pd.DataFrame:
    order = _parse_dates(raw["draw_date"]).argsort(kind="stable")
    return raw.iloc[order.to_numpy()].reset_index(drop=True)


def ensure_store(path: Path | None = None) -> Path:
    path = _p(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}. Run the scraper or upload a results file.")
    return path


def load_store(path: Path | None = None) -> pd.DataFrame:
    """Validated draws (draw_number, draw_date, n1..n6, additional), oldest first."""
    return ingest_file(ensure_store(path)).draws


def validate_new_draw(draws: pd.DataFrame, draw_date: date, numbers: list[int], additional: int) -> list[str]:
    """Return a list of problems (empty list = OK)."""
    errs = []
    nums = [int(n) for n in numbers]
    if len(nums) != PICK:
        errs.append(f"Select exactly {PICK} main numbers (got {len(nums)}).")
    if len(set(nums)) != len(nums):
        errs.append("Main numbers must be different.")
    if any(not 1 <= n <= N_NUMBERS for n in nums + [int(additional)]):
        errs.append(f"Numbers must be between 1 and {N_NUMBERS}.")
    if int(additional) in nums:
        errs.append("The additional number cannot be one of the main numbers.")
    d = pd.Timestamp(draw_date)
    existing = pd.to_datetime(draws["draw_date"])
    if (existing == d).any():
        errs.append(f"A draw on {d.date()} already exists.")
    elif len(existing) and d < existing.max():
        errs.append(f"{d.date()} is before the latest stored draw ({existing.max().date()}). "
                    "Upload a file to insert older draws.")
    return errs


def _new_row(raw: pd.DataFrame, fmt: str, d, nums, additional) -> dict:
    row = {c: "" for c in raw.columns}
    row["draw_date"] = pd.Timestamp(d).strftime(fmt)
    for c, n in zip(MAIN_COLS, sorted(int(x) for x in nums)):
        row[c] = str(n)
    row["additional"] = str(int(additional))
    return row


def add_draw(draw_date: date, numbers: list[int], additional: int, path: Path | None = None) -> pd.DataFrame:
    path = _p(path)
    draws = load_store(path)
    errs = validate_new_draw(draws, draw_date, numbers, additional)
    if errs:
        raise ValueError(" ".join(errs))
    raw = _read_raw(path)
    raw = pd.concat([raw, pd.DataFrame([_new_row(raw, _date_format(raw), draw_date, numbers, additional)])],
                    ignore_index=True)
    _write_raw(raw, path)
    return load_store(path)


def merge_upload(content: bytes, filename: str, path: Path | None = None) -> dict:
    """Append draws from a CSV/XLSX whose dates are not stored yet. Existing rows are never changed."""
    path = _p(path)
    draws = load_store(path)
    new, report, _ = validate_and_clean(read_table(content, filename))
    have = set(pd.to_datetime(draws["draw_date"]))
    new_dates = pd.to_datetime(new["draw_date"])
    fresh = new[~new_dates.isin(have)]
    conflicts = []
    both = new[new_dates.isin(have)].merge(draws[REQUIRED_COLS], on="draw_date", suffixes=("", "_stored"))
    for _, r in both.iterrows():
        cols = MAIN_COLS + ["additional"]
        if [r[c] for c in cols] != [r[f"{c}_stored"] for c in cols]:
            conflicts.append(pd.Timestamp(r["draw_date"]).strftime("%Y-%m-%d"))
    if len(fresh):
        raw = _read_raw(path)
        fmt = _date_format(raw)
        rows = [_new_row(raw, fmt, r["draw_date"], [r[c] for c in MAIN_COLS], r["additional"])
                for _, r in fresh.iterrows()]
        _write_raw(_sort_raw(pd.concat([raw, pd.DataFrame(rows)], ignore_index=True)), path)
    return {"added": int(len(fresh)), "already_present": int(len(new) - len(fresh)),
            "conflicts_kept_stored_version": conflicts, "file_report": report}


def delete_last(path: Path | None = None) -> pd.Series:
    """Remove the most recent draw (by date) from the file."""
    path = _p(path)
    draws = load_store(path)
    if len(draws) <= 1:
        raise ValueError("Nothing to delete.")
    raw = _read_raw(path)
    dates = _parse_dates(raw["draw_date"])
    _write_raw(raw.drop(index=dates.idxmax()).reset_index(drop=True), path)
    return draws.iloc[-1]
