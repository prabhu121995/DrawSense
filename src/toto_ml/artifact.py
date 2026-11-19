"""Best-model selection, artifact persistence and next-draw prediction.

The notebook (notebook.ipynb) trains & benchmarks every algorithm, calls
``select_best`` and ``save_artifact``. The web UI / CLI only call
``load_artifact`` and ``predict_next``; they never train.

Artifact folder layout (default ``artifacts/best_model/``):
    model.joblib      fitted scorer(s) + calibrator + config (everything needed to predict)
    metadata.json     selection rule, chosen model, leaderboards, significance, training range
    leaderboard.csv   validation & test metrics for all algorithms
    history.csv       the draws the model was trained on (draw_date, n1..n6, additional)
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .config import BASELINE_P, PROJECT_ROOT, config_hash
from .data.schema import REQUIRED_COLS
from .evaluation.metrics import EXPECTED
from .features import build_features
from .models import LABELS, make_scorer, rank_mean_ensemble
from .prediction import JACKPOT_ODDS, number_history_stats, rank_numbers, ranked_sets

DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "best_model"
SELECTION_METRIC = "log_loss"   # proper scoring rule over all 49 probabilities: least noisy choice
SELECTION_SPLIT = "validation"  # never select on the final test period


def leaderboard_frame(res) -> pd.DataFrame:
    rows = []
    for split, items in res.leaderboard.items():
        for r in items:
            rows.append({"split": split, "model": r["model"], "label": LABELS.get(r["model"], r["model"]), **r})
    df = pd.DataFrame(rows)
    sig = res.significance["models"]
    df["p_hits10_holm"] = [sig.get(m, {}).get("p_hits10_holm") if s == "test" else None
                           for m, s in zip(df["model"], df["split"])]
    return df


def select_best(res, metric: str = SELECTION_METRIC, split: str = SELECTION_SPLIT) -> tuple[str, pd.DataFrame]:
    """Pick the best non-random model on the development (validation) period only."""
    lb = leaderboard_frame(res)
    cand = lb[(lb["split"] == split) & (lb["model"] != "random")].copy()
    ascending = metric in ("log_loss", "brier", "ece")
    cand = cand.sort_values([metric, "hits@10"], ascending=[ascending, False]).reset_index(drop=True)
    return str(cand.loc[0, "model"]), cand


def save_artifact(res, best: str, draws: pd.DataFrame, out_dir: str | Path = DEFAULT_ARTIFACT_DIR,
                  selection_table: pd.DataFrame | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    o = res.outputs[best]
    members = o.params.get("members") if best == "ensemble" else None
    scorers = ({m: res.outputs[m].production_scorer for m in members} if members
               else {best: o.production_scorer})
    sp = res.split
    dates = pd.to_datetime(draws["draw_date"])
    sig = res.significance["models"].get(best, {})
    test_row = next(r for r in res.leaderboard["test"] if r["model"] == best)
    val_row = next(r for r in res.leaderboard["validation"] if r["model"] == best)
    rnd_test = next((r for r in res.leaderboard["test"] if r["model"] == "random"), None)
    meta = {
        "artifact_version": __version__,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_hash": res.config_hash,
        "best_model": best,
        "best_model_label": LABELS.get(best, best),
        "ensemble_members": members,
        "selection": {"metric": SELECTION_METRIC, "split": SELECTION_SPLIT,
                      "rule": f"lowest {SELECTION_METRIC} on the development walk-forward period "
                              "(random baseline excluded); the final test period is NOT used for selection"},
        "training_data": {"n_draws": int(len(draws)), "date_min": dates.min().strftime("%Y-%m-%d"),
                          "date_max": dates.max().strftime("%Y-%m-%d")},
        "validation_period": {"from": dates.iloc[sp.val_start].strftime("%Y-%m-%d"),
                              "to": dates.iloc[sp.test_start - 1].strftime("%Y-%m-%d"),
                              "n_draws": int(len(sp.val_states))},
        "test_period": {"from": dates.iloc[sp.test_start].strftime("%Y-%m-%d"),
                        "to": dates.iloc[sp.T - 1].strftime("%Y-%m-%d"), "n_draws": int(len(sp.test_states))},
        "validation_metrics": {k: val_row[k] for k in ("hits@6", "hits@10", "log_loss", "brier", "roc_auc", "ndcg@10")},
        "test_metrics": {k: test_row[k] for k in ("hits@6", "hits@10", "log_loss", "brier", "roc_auc", "ndcg@10",
                                                  "top6_2plus_rate", "top6_3plus_rate")},
        "random_test_metrics": {k: rnd_test[k] for k in ("hits@6", "hits@10")} if rnd_test else None,
        "expected_random": {"hits@6": EXPECTED["hits@6"], "hits@10": EXPECTED["hits@10"],
                            "log_loss": EXPECTED["log_loss"], "marginal_probability": BASELINE_P},
        "test_significance": {
            "p_hits10_vs_random": sig.get("p_hits10_vs_random_sim"),
            "p_hits10_holm": sig.get("p_hits10_holm"),
            "significant": bool(sig.get("significant_hits10", False)),
            "verdict": sig.get("verdict"),
            "summary": res.significance["summary"],
        },
        "feature_names": res.features.feature_names,
    }
    bundle = {"model_name": best, "members": members, "scorers": scorers,
              "calibrator": o.production_calibrator, "config": res.cfg, "metadata": meta}
    joblib.dump(bundle, out / "model.joblib", compress=3)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    leaderboard_frame(res).to_csv(out / "leaderboard.csv", index=False)
    if selection_table is not None:
        selection_table.to_csv(out / "selection.csv", index=False)
    hist = draws.copy()
    hist["draw_date"] = pd.to_datetime(hist["draw_date"]).dt.strftime("%Y-%m-%d")
    hist[REQUIRED_COLS].to_csv(out / "history.csv", index=False)
    return out


def load_artifact(path: str | Path = DEFAULT_ARTIFACT_DIR) -> dict:
    path = Path(path)
    f = path / "model.joblib" if path.is_dir() else path
    if not f.exists():
        raise FileNotFoundError(f"No model artifact at {f}. Run notebook.ipynb first to train and export it.")
    bundle = joblib.load(f)
    bundle["dir"] = f.parent
    return bundle


def refit(bundle: dict, draws: pd.DataFrame, out_dir: str | Path | None = None) -> dict:
    """Quick update after new draws: refit the SAME chosen algorithm (same settings, same seed) on all draws.

    No re-benchmarking and no algorithm re-selection (that is what notebook.ipynb does). The probability
    calibrator and the test evidence are kept from the last full notebook run. Takes seconds, not minutes.
    """
    out = Path(out_dir or bundle.get("dir") or DEFAULT_ARTIFACT_DIR)
    cfg = bundle["config"]
    meta = dict(bundle["metadata"])
    fs = build_features(draws, cfg)
    if fs.feature_names != meta["feature_names"]:
        raise ValueError("Feature set changed since the last notebook run. Run the full notebook instead.")
    labelled = fs.frame[fs.frame["t"] < fs.T]
    scorers = {}
    for name, sc in bundle["scorers"].items():
        if getattr(sc, "needs_fit", False):
            scorers[name] = make_scorer(name, cfg, fs.feature_names, sc.params).fit(labelled)
        else:  # frequency-type baselines have nothing to fit; they read the features directly
            scorers[name] = sc
    dates = pd.to_datetime(draws["draw_date"])
    prev_n = meta["training_data"]["n_draws"]
    meta.setdefault("benchmark_created_at", meta["created_at"])
    meta["created_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta["training_data"] = {"n_draws": int(len(draws)), "date_min": dates.min().strftime("%Y-%m-%d"),
                             "date_max": dates.max().strftime("%Y-%m-%d")}
    meta["last_update"] = {"type": "refit", "previous_n_draws": prev_n, "new_draws": int(len(draws) - prev_n),
                           "note": f"{meta['best_model_label']} refitted on all draws with the same settings. "
                                   "Algorithm choice and test evidence are from the last full notebook run "
                                   f"({meta['benchmark_created_at']})."}
    new = {k: v for k, v in bundle.items() if k != "dir"}
    new.update(scorers=scorers, metadata=meta)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "model.joblib.tmp"
    joblib.dump(new, tmp, compress=3)
    tmp.replace(out / "model.joblib")
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    hist = draws.copy()
    hist["draw_date"] = dates.dt.strftime("%Y-%m-%d")
    hist[REQUIRED_COLS].to_csv(out / "history.csv", index=False)
    new["dir"] = out
    return new


def load_history(bundle: dict) -> pd.DataFrame:
    from .data import ingest_file
    return ingest_file(bundle["dir"] / "history.csv").draws


def next_draw_date(last: date) -> date:
    """Next regular TOTO draw: Monday or Thursday (special draws are not modelled)."""
    d = last + timedelta(days=1)
    while d.weekday() not in (0, 3):
        d += timedelta(days=1)
    return d


def predict_next(bundle: dict, draws: pd.DataFrame | None = None, n_sets: int = 5, pool: int = 12,
                 max_overlap: int = 3, window: int = 20) -> dict:
    """Score all 49 numbers for the draw after the last one in ``draws`` and build ranked sets."""
    draws = load_history(bundle) if draws is None else draws
    cfg = bundle["config"]
    fs = build_features(draws, cfg)
    live = fs.rows_for_states([fs.T])
    if bundle["members"]:
        scores = rank_mean_ensemble({m: s.score(live) for m, s in bundle["scorers"].items()}, live["t"].to_numpy())
    else:
        scores = bundle["scorers"][bundle["model_name"]].score(live)
    probs = bundle["calibrator"].transform(scores)
    seed = int(cfg["seed"])
    order = rank_numbers(probs, scores, seed)
    hist = number_history_stats(draws, window).set_index("number")
    numbers = []
    for r, i in enumerate(order, start=1):
        n = int(i) + 1
        h = hist.loc[n]
        numbers.append({"rank": r, "number": n, "probability": float(probs[i]), "score": float(scores[i]),
                        "hist_freq": float(h["hist_freq"]), "recent_count": int(h["recent_count"]),
                        "recent_window": int(h["recent_window"]), "last_seen_date": h["last_seen_date"],
                        "draws_since_last": None if pd.isna(h["draws_since_last"]) else int(h["draws_since_last"])})
    last = pd.to_datetime(draws["draw_date"]).iloc[-1].date()
    last_row = draws.iloc[-1]
    meta = bundle["metadata"]
    return {
        "model": meta["best_model_label"],
        "model_id": bundle["model_name"],
        "based_on": {"n_draws": int(len(draws)), "last_draw_date": last.isoformat(),
                     "last_numbers": [int(last_row[f"n{k}"]) for k in range(1, 7)],
                     "last_additional": int(last_row["additional"])},
        "next_draw_date": next_draw_date(last).isoformat(),
        "sets": ranked_sets(probs, scores, n_sets=n_sets, pool=pool, max_overlap=max_overlap, seed=seed),
        "numbers": numbers,
        "prob_sum": float(probs.sum()),
        "evidence": meta["test_significance"],
        "disclaimer": (f"TOTO draws are random. Every set has the same 1 in {JACKPOT_ODDS:,} chance of winning "
                       "Group 1; rankings are model estimates, not better odds."),
        "config_hash": config_hash(cfg),
    }
