"""Best-model selection, artifact persistence and next-draw prediction.

The notebook (notebook.ipynb) trains & benchmarks every algorithm, calls
``select_best`` and ``save_artifact``. The web UI / CLI only call
``load_artifact`` and ``predict_next``; they never train.

Artifact folder layout (default ``artifacts/best_model/``):
    model.joblib      fitted scorers + calibrators + config (everything needed to predict):
                        scorers / calibrator              trained on 100% of draws (refitted after new draws)
                        split_scorers / split_calibrator  trained on the development split only (before the
                                                          final test period), frozen until the next notebook run
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
from .value import fit_popularity, number_popularity, shape_profile, value_sets

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
    names = members or [best]
    scorers = {m: res.outputs[m].production_scorer for m in names}   # 100% of draws
    split_scorers = {m: res.outputs[m].final_scorer for m in names}  # development split only (tested model)
    label = (f"Ensemble ({' + '.join(LABELS.get(m, m) for m in members)})" if members
             else LABELS.get(best, best))
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
        "best_model_label": label,
        "ensemble_members": members,
        "selection": {"metric": SELECTION_METRIC, "split": SELECTION_SPLIT,
                      "rule": ("a fixed choice in configs/default.yaml (live.model); the final test period is "
                               "only used to report its result" if best == res.cfg.get("live", {}).get("model") else
                               f"lowest {SELECTION_METRIC} on the development walk-forward period "
                               "(random baseline excluded); the final test period is NOT used for selection")},
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
              "calibrator": o.production_calibrator, "split_scorers": split_scorers,
              "split_calibrator": o.calibrator, "popularity": fit_popularity(), "config": res.cfg, "metadata": meta}
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

    Only the 100% model is refitted. The split model stays as trained on the development split, so its
    sets keep matching the test evidence; it still scores the next draw from the latest history.

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
    new.update(scorers=scorers, metadata=meta, popularity=fit_popularity())
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


def _score(scorers: dict, calibrator, live: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Raw scores (rank-mean over members for an ensemble) and calibrated probabilities for 49 numbers."""
    if len(scorers) > 1:
        scores = rank_mean_ensemble({m: s.score(live) for m, s in scorers.items()}, live["t"].to_numpy())
    else:
        scores = next(iter(scorers.values())).score(live)
    return scores, calibrator.transform(scores)


def predict_next(bundle: dict, draws: pd.DataFrame | None = None, n_split: int | None = None,
                 n_full: int | None = None, pool: int | None = None, max_overlap: int = 3, window: int = 20,
                 strategy: str | None = None) -> dict:
    """Score all 49 numbers for the draw after the last one in ``draws`` and build ranked sets.

    ``n_split`` sets come from the model trained on the development split, then ``n_full`` sets from the
    model trained on 100% of draws (defaults: ``live`` in the config, 2 + 2). A full-data set shares at
    most ``max_overlap`` numbers with any set before it. Artifacts without a split model give full sets only.

    ``strategy`` (config ``live.strategy``): "value" builds sets from the model's top ``pool`` numbers in a
    typical draw shape, ordered by prize value (fewer players share a win); "model" uses the model ranking only.
    """
    draws = load_history(bundle) if draws is None else draws
    cfg = bundle["config"]
    live_cfg = cfg.get("live", {})
    n_split = int(live_cfg.get("sets_split", 2) if n_split is None else n_split)
    n_full = int(live_cfg.get("sets_full", 2) if n_full is None else n_full)
    if not bundle.get("split_scorers"):
        n_full, n_split = n_full + n_split, 0
    strategy = strategy or live_cfg.get("strategy", "value")
    pool = int(pool or live_cfg.get("pool", 18 if strategy == "value" else 12))
    pop = bundle.get("popularity") or fit_popularity()
    prof = shape_profile(draws)

    def build(p, sc, n, earlier):
        if strategy == "value":
            return value_sets(p, sc, pop, prof, n_sets=n, pool=pool, max_overlap=max_overlap, seed=seed,
                              earlier=earlier)
        return ranked_sets(p, sc, n_sets=n, pool=pool, max_overlap=max_overlap, seed=seed, earlier=earlier)
    fs = build_features(draws, cfg)
    live = fs.rows_for_states([fs.T])
    seed = int(cfg["seed"])
    scores, probs = _score(bundle["scorers"], bundle["calibrator"], live)
    sets = []
    if n_split:
        s_scores, s_probs = _score(bundle["split_scorers"], bundle["split_calibrator"], live)
        sets += [{**s, "source": "split"} for s in build(s_probs, s_scores, n_split, [])]
    if n_full:
        sets += [{**s, "source": "full"} for s in build(probs, scores, n_full, [x["numbers"] for x in sets])]
    for r, s in enumerate(sets, start=1):
        s["source_rank"] = s["rank"]
        s["rank"] = r
    order = rank_numbers(probs, scores, seed)
    hist = number_history_stats(draws, window).set_index("number")
    popn = number_popularity(pop).set_index("number")
    numbers = []
    for r, i in enumerate(order, start=1):
        n = int(i) + 1
        h = hist.loc[n]
        numbers.append({"rank": r, "number": n, "probability": float(probs[i]), "score": float(scores[i]),
                        "hist_freq": float(h["hist_freq"]), "recent_count": int(h["recent_count"]),
                        "recent_window": int(h["recent_window"]), "last_seen_date": h["last_seen_date"],
                        "value_pct": float(popn.loc[n, "value_pct"]),
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
        "sets": sets,
        "strategy": strategy,
        "popularity": {k: v for k, v in pop.items() if k != "weights"},
        "shape": prof,
        "sources": {"split": "trained on the development split (draws before the final test), the tested model",
                    "full": "trained on 100% of draws (refitted when new draws are added)"},
        "numbers": numbers,
        "prob_sum": float(probs.sum()),
        "evidence": meta["test_significance"],
        "disclaimer": (f"Fair play verified: every set has the same 1 in {JACKPOT_ODDS:,} chance of winning Group 1. "
                       "DrawSense builds sets for prize value, so a winning set shares its prize with fewer people."),
        "config_hash": config_hash(cfg),
    }
