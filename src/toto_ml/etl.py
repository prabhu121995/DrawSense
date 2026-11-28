"""ETL: get the latest official TOTO and 4D results, store both in drawsense.db, update the model and next move.

    extract    toto_scraper.py  -> toto_results.csv   (incremental: only draws after the last one in the file)
               fourd_scraper.py -> fourd_results.csv  (incremental, same way)
    transform  TOTO: merge rows for the same date: the official scraped row (draw number + prize data)
               replaces a row typed in by hand. Both games: Axis & Mirror metrics for every number.
    load       both CSV files -> drawsense.db (one database, filter by game)
               TOTO only: quick refit of the 100% data model -> artifacts/next_move.csv
               (the prediction algorithm is TOTO-only; 4D is stored and analysed separately)

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
FOURD_SCRAPER = PROJECT_ROOT / "fourd_scraper.py"


def _run_script(script: Path, out: Path, say: Callable[[str], None], timeout: int) -> tuple[int, list[str]]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, "-u", str(script), "-o", str(out)], cwd=str(PROJECT_ROOT),
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
    return code, lines


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
            update_model: bool = True, timeout: int = 1800, fourd: bool = True,
            fourd_path: str | Path | None = None, db_path: str | Path | None = None) -> dict:
    from . import db
    from .store import STORE_PATH, backup_path, load_store
    path = Path(path or STORE_PATH)
    fourd_path = Path(fourd_path or db.FOURD_PATH)
    say = on_line or (lambda s: None)
    before = load_store(path) if path.exists() else None
    n_before = 0 if before is None else len(before)
    if path.exists() and not backup_path(path).exists():
        shutil.copy2(path, backup_path(path))  # same one-time safety copy as the app's own edits

    started = db.now_iso()
    say("TOTO: checking Singapore Pools for new results…")
    code, lines = _run_script(SCRAPER, path, say, timeout)
    merged = _merge_same_dates(path) if path.exists() else 0
    after = load_store(path)
    added = len(after) - n_before
    result = {"ok": code == 0, "returncode": code, "added": added, "merged_manual_rows": merged,
              "n_draws": len(after), "last_draw_date": pd.to_datetime(after["draw_date"]).iloc[-1].strftime("%Y-%m-%d"),
              "log": lines[-60:], "model_updated": False, "next_move_updated": False,
              "fourd": {"ok": None, "added": 0, "n_draws": 0, "last_draw_date": None}, "db": None}

    fd_started = db.now_iso()
    if fourd:
        fd_before = len(db.read_fourd_csv(fourd_path))
        say("4D: checking Singapore Pools for new results…")
        fd_code, fd_lines = _run_script(FOURD_SCRAPER, fourd_path, say, timeout)
        fd = db.read_fourd_csv(fourd_path)
        last = pd.to_datetime(fd["draw_date"], format="%a, %d %b %Y", errors="coerce").max() if len(fd) else None
        result["fourd"] = {"ok": fd_code == 0, "added": len(fd) - fd_before, "n_draws": len(fd),
                           "last_draw_date": None if last is None or pd.isna(last) else last.strftime("%Y-%m-%d")}
        result["log"] = (lines + fd_lines)[-80:]

    say("Saving both games to the database…")
    result["db"] = {"path": str(db.DB_PATH if db_path is None else db_path),
                    "draws": db.sync_all(path, fourd_path, db_path)}
    db.log_etl_run("toto", started, result["ok"], added, len(after), result["last_draw_date"],
                   f"merged {merged} hand-typed row(s)" if merged else "", db_path)
    if fourd:
        f = result["fourd"]
        db.log_etl_run("4d", fd_started, f["ok"], f["added"], f["n_draws"], f["last_draw_date"], "", db_path)

    if update_model:
        from .artifact import DEFAULT_ARTIFACT_DIR, load_artifact, refit
        from .nextmove import load_next_move, is_current, refresh_next_move
        if (DEFAULT_ARTIFACT_DIR / "model.joblib").exists():
            bundle = load_artifact(DEFAULT_ARTIFACT_DIR)
            trained = bundle["metadata"]["training_data"]
            if added or merged or trained["n_draws"] != len(after):
                say("Updating the TOTO model with the new results…")
                bundle = refit(bundle, after)
                result["model_updated"] = True
            _, meta = load_next_move()
            if result["model_updated"] or not is_current(meta, after):
                say("Building your next move…")
                refresh_next_move(bundle, after)
                result["next_move_updated"] = True
    say("Done.")
    return result
