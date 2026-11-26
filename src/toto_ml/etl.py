"""ETL: get the latest official results from the web, update toto_results.csv, the model and the next move.

    extract    toto_scraper.py (incremental: only draws after the last one in the file)
    transform  merge rows for the same date: the official scraped row (draw number + prize data)
               replaces a row typed in by hand
    load       toto_results.csv -> quick refit of the 100% data model -> artifacts/next_move.csv

Run from the app ("Get latest results"), or:  uv run toto-ml etl
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import PROJECT_ROOT

SCRAPER = PROJECT_ROOT / "toto_scraper.py"


def _merge_same_dates(path: Path) -> int:
    """Keep one row per draw date, preferring the scraped row (it has a draw number). Returns rows removed."""
    from .data.ingest import _parse_dates
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if raw.empty:
        return 0
    dates = _parse_dates(raw["draw_date"])
    official = raw["draw_number"] if "draw_number" in raw else pd.Series("", index=raw.index)
    raw = raw.assign(_d=dates, _official=official.astype(str).str.strip().ne(""))
    raw = raw.sort_values(["_d", "_official"], ascending=[True, False], kind="stable")
    keep = ~raw["_d"].duplicated(keep="first") | raw["_d"].isna()
    removed = int((~keep).sum())
    out = raw[keep].drop(columns=["_d", "_official"])
    if removed:
        tmp = path.with_suffix(".tmp")
        out.to_csv(tmp, index=False)
        tmp.replace(path)
    return removed


def run_etl(path: str | Path | None = None, on_line: Callable[[str], None] | None = None,
            update_model: bool = True, timeout: int = 1800) -> dict:
    from .store import STORE_PATH, backup_path, load_store
    path = Path(path or STORE_PATH)
    say = on_line or (lambda s: None)
    before = load_store(path) if path.exists() else None
    n_before = 0 if before is None else len(before)
    if path.exists() and not backup_path(path).exists():
        shutil.copy2(path, backup_path(path))  # same one-time safety copy as the app's own edits

    say("Checking Singapore Pools for new results…")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, "-u", str(SCRAPER), "-o", str(path)], cwd=str(PROJECT_ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env=env)
    lines = []
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                lines.append(line)
                say(line)
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        code = -1
        lines.append("Stopped: the website took too long to answer.")
    merged = _merge_same_dates(path) if path.exists() else 0
    after = load_store(path)
    added = len(after) - n_before
    result = {"ok": code == 0, "returncode": code, "added": added, "merged_manual_rows": merged,
              "n_draws": len(after), "last_draw_date": pd.to_datetime(after["draw_date"]).iloc[-1].strftime("%Y-%m-%d"),
              "log": lines[-60:], "model_updated": False, "next_move_updated": False}

    if update_model:
        from .artifact import DEFAULT_ARTIFACT_DIR, load_artifact, refit
        from .nextmove import load_next_move, is_current, refresh_next_move
        if (DEFAULT_ARTIFACT_DIR / "model.joblib").exists():
            bundle = load_artifact(DEFAULT_ARTIFACT_DIR)
            trained = bundle["metadata"]["training_data"]
            if added or merged or trained["n_draws"] != len(after):
                say("Updating the model with the new results…")
                bundle = refit(bundle, after)
                result["model_updated"] = True
            _, meta = load_next_move()
            if result["model_updated"] or not is_current(meta, after):
                say("Building your next move…")
                refresh_next_move(bundle, after)
                result["next_move_updated"] = True
    say("Done.")
    return result
