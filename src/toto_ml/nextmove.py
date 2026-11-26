"""The stored "next move": the sets for the next draw, saved once and shown instantly by the app.

Written after every model update (Save draw, Get latest results, Update model, full benchmark):
    artifacts/next_move.csv    one row per set (what the app shows and the user downloads)
    artifacts/next_move.json   draw date, history it was built from, model, prize-value summary

The app reads these files instead of recomputing on every page load. They are rebuilt only when the
results file has a draw the saved move does not know yet.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT

NEXT_MOVE_CSV = PROJECT_ROOT / "artifacts" / "next_move.csv"
NEXT_MOVE_JSON = NEXT_MOVE_CSV.with_suffix(".json")
SOURCE_LABEL = {"split": "Validated model", "full": "Full-history model"}


def save_next_move(pred: dict, csv_path: str | Path | None = None) -> Path:
    csv_path = Path(csv_path or NEXT_MOVE_CSV)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"rank": s["rank"], "model": SOURCE_LABEL.get(s["source"], s["source"]),
             **{f"n{i + 1}": n for i, n in enumerate(s["numbers"])},
             "prize_value_pct": round(s.get("value_pct", 0.0), 1), "typical_shape": s.get("typical_shape", ""),
             "odd": s["odd"], "even": s["even"], "sum": s["sum"],
             "avg_probability": round(s["mean_probability"], 6)} for s in pred["sets"]]
    tmp = csv_path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(csv_path)
    meta = {k: pred[k] for k in ("model", "model_id", "next_draw_date", "based_on", "strategy", "popularity", "shape",
                                 "disclaimer") if k in pred}
    meta["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["top_numbers"] = [{"rank": x["rank"], "number": x["number"], "value_pct": round(x.get("value_pct", 0), 1)}
                           for x in pred.get("numbers", [])]
    csv_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return csv_path


def load_next_move(csv_path: str | Path | None = None) -> tuple[pd.DataFrame | None, dict | None]:
    csv_path = Path(csv_path or NEXT_MOVE_CSV)
    js = csv_path.with_suffix(".json")
    if not csv_path.exists() or not js.exists():
        return None, None
    try:
        return pd.read_csv(csv_path), json.loads(js.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None


def is_current(meta: dict | None, draws: pd.DataFrame) -> bool:
    """True when the saved move was built from exactly the draws in the results file."""
    if not meta:
        return False
    last = pd.to_datetime(draws["draw_date"]).iloc[-1].strftime("%Y-%m-%d")
    b = meta.get("based_on", {})
    return b.get("last_draw_date") == last and int(b.get("n_draws", -1)) == len(draws)


def refresh_next_move(bundle: dict, draws: pd.DataFrame, csv_path: str | Path | None = None) -> dict:
    from .artifact import predict_next
    pred = predict_next(bundle, draws)
    save_next_move(pred, csv_path)
    return pred
