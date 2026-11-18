"""DrawSense: TOTO next-draw ranked sets (Streamlit app).

Run:  uv run streamlit run app.py      (opens http://localhost:8501)

* Add results - save a new draw to toto_results.csv -> retrain (runs notebook.ipynb) -> next 5 sets
* Next sets   - ranked six-number sets from the trained model
* Model       - which algorithm was chosen, its test evidence, manual retrain
"""
from __future__ import annotations

import subprocess
import sys
import time

import pandas as pd
import streamlit as st

from toto_ml.artifact import DEFAULT_ARTIFACT_DIR, load_artifact, next_draw_date, predict_next
from toto_ml.config import PROJECT_ROOT
from toto_ml.store import STORE_PATH, add_draw, backup_path, delete_last, load_store, merge_upload, validate_new_draw

NOTEBOOK = PROJECT_ROOT / "notebook.ipynb"
MODEL_FILE = DEFAULT_ARTIFACT_DIR / "model.joblib"

st.set_page_config(page_title="DrawSense", layout="centered")
st.markdown("""<style>
.ball{display:inline-flex;align-items:center;justify-content:center;width:42px;height:42px;border-radius:50%;
  background:#2a78d6;color:#fff;font-weight:700;font-size:17px;margin:2px 4px 2px 0;font-variant-numeric:tabular-nums}
.ball.sm{width:28px;height:28px;font-size:12px}
.ball.add{background:transparent;color:inherit;border:2px dashed #2a78d6}
.setrow{display:flex;align-items:center;gap:14px;padding:8px 0;border-bottom:1px solid rgba(128,128,128,.25);flex-wrap:wrap}
.setrank{font-size:22px;font-weight:700;width:46px;opacity:.8}
.setmeta{font-size:13px;opacity:.75;margin-left:auto;text-align:right}
</style>""", unsafe_allow_html=True)


def balls(nums, cls=""):
    return "".join(f'<span class="ball {cls}">{n}</span>' for n in nums)


@st.cache_resource(show_spinner=False)
def get_bundle(mtime: float):
    return load_artifact(DEFAULT_ARTIFACT_DIR)


def bundle_or_none():
    if not MODEL_FILE.exists():
        return None
    return get_bundle(MODEL_FILE.stat().st_mtime)  # reloads automatically after retraining


def run_notebook() -> tuple[bool, str]:
    """Retrain = execute notebook.ipynb (training, algorithm selection, artifact export)."""
    with st.status("Retraining: running notebook.ipynb (about 3 minutes)…", expanded=False) as s:
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                               "--inplace", "--ExecutePreprocessor.timeout=3600", str(NOTEBOOK)],
                              cwd=PROJECT_ROOT, capture_output=True, text=True)
        if proc.returncode == 0:
            s.update(label=f"Retrained in {time.time() - t0:.0f}s", state="complete")
            st.cache_resource.clear()
            return True, ""
        s.update(label="Retraining failed", state="error")
        return False, proc.stderr[-3000:]


def show_sets(bundle, draws, n_sets: int = 5, key: str = "sets"):
    r = predict_next(bundle, draws, n_sets=int(n_sets))
    last = r["based_on"]
    st.subheader(f"Next draw: {pd.Timestamp(r['next_draw_date']):%a %d %b %Y}")
    st.markdown(f"Model **{r['model']}** · based on {last['n_draws']} draws · last result "
                f"{pd.Timestamp(last['last_draw_date']):%d %b %Y}: {balls(last['last_numbers'], 'sm')} + "
                f"{balls([last['last_additional']], 'sm add')}", unsafe_allow_html=True)
    st.markdown("".join(
        f'<div class="setrow"><div class="setrank">#{s["rank"]}</div><div>{balls(s["numbers"])}</div>'
        f'<div class="setmeta">avg probability {s["mean_probability"]:.3%}<br>number ranks '
        f'{", ".join(map(str, s["number_ranks"]))}<br>{s["odd"]} odd / {s["even"]} even · sum {s["sum"]}</div></div>'
        for s in r["sets"]), unsafe_allow_html=True)
    csv = pd.DataFrame([{"rank": s["rank"], **{f"n{i + 1}": n for i, n in enumerate(s["numbers"])},
                         "avg_probability": round(s["mean_probability"], 6)} for s in r["sets"]])
    st.download_button("Download sets (CSV)", csv.to_csv(index=False), f"drawsense_sets_{r['next_draw_date']}.csv",
                       "text/csv", key=f"dl_{key}")
    return r


st.title("DrawSense")
st.caption("Singapore TOTO · next-draw ranked sets")
draws = load_store()
bundle = bundle_or_none()
trained_to = pd.Timestamp(bundle["metadata"]["training_data"]["date_max"]) if bundle else None
n_untrained = int((pd.to_datetime(draws["draw_date"]) > trained_to).sum()) if bundle else len(draws)

# Result of the last "save -> retrain" run, kept across the rerun
if "flash" in st.session_state:
    kind, msg = st.session_state.pop("flash")
    getattr(st, kind)(msg)

tab_add, tab_sets, tab_model = st.tabs(["Add results", "Next sets", "Model"])

# ------------------------------------------------------------------- add results
with tab_add:
    last_date = pd.to_datetime(draws["draw_date"]).max().date()
    st.markdown(f"**{len(draws)} draws** in `{STORE_PATH.name}` "
                f"({pd.to_datetime(draws['draw_date']).min():%d %b %Y} → {last_date:%d %b %Y})")

    with st.form("add_draw", clear_on_submit=True):
        st.subheader("Add a new draw result")
        d = st.date_input("Draw date", value=next_draw_date(last_date), format="DD/MM/YYYY")
        main = st.multiselect("Six winning numbers", list(range(1, 50)), max_selections=6)
        add = st.number_input("Additional number", 1, 49, value=None, placeholder="1–49")
        auto = st.checkbox("Retrain and predict the next 5 sets after saving", value=True)
        submitted = st.form_submit_button("Save draw", type="primary")
    if submitted:
        errs = validate_new_draw(draws, d, main, add) if add else ["Enter the additional number."]
        if errs:
            for e in errs:
                st.error(e)
        else:
            draws = add_draw(d, main, add)
            st.success(f"Saved to {STORE_PATH.name}: {d:%a %d %b %Y} · {sorted(main)} + {add}")
            if auto:
                ok, err = run_notebook()
                if ok:
                    bundle = bundle_or_none()
                    st.success(f"Model retrained on {len(draws)} draws "
                               f"(best algorithm: {bundle['metadata']['best_model_label']}).")
                    show_sets(bundle, draws, 5, key="after_add")
                else:
                    st.error("Saved, but retraining failed. See details below; your draw is kept.")
                    st.code(err)
            else:
                st.session_state["flash"] = ("success", f"Saved {d:%d %b %Y}. Retrain later in the Model tab.")
                st.rerun()

    st.subheader("Or add from a file")
    up = st.file_uploader("CSV or XLSX with draw_date, n1..n6, additional (only new dates are added)",
                          type=["csv", "xlsx"])
    if up is not None and st.button("Add new draws from file"):
        try:
            res = merge_upload(up.getvalue(), up.name)
            st.success(f"Added {res['added']} new draw(s) to {STORE_PATH.name}; "
                       f"{res['already_present']} already stored.")
            if res["conflicts_kept_stored_version"]:
                st.warning("Different numbers for already-stored dates (stored version kept): "
                           + ", ".join(res["conflicts_kept_stored_version"]))
            for row in res["file_report"]["excluded_rows"][:10]:
                st.caption(f"Skipped row {row['row']}: {row['reason']}")
            if res["added"]:
                st.info("Retrain in the Model tab so the model learns from the new draws.")
        except ValueError as e:
            st.error(str(e))

    st.subheader("Latest draws")
    show = draws.tail(10).iloc[::-1].copy()
    show["draw_date"] = pd.to_datetime(show["draw_date"]).dt.strftime("%a %d %b %Y")
    st.dataframe(show[["draw_date", "n1", "n2", "n3", "n4", "n5", "n6", "additional"]], hide_index=True,
                 use_container_width=True)
    with st.expander("Fix a mistake"):
        lr = draws.iloc[-1]
        st.write(f"Delete the latest draw ({pd.Timestamp(lr['draw_date']):%d %b %Y}: "
                 f"{[int(lr[f'n{i}']) for i in range(1, 7)]} + {int(lr['additional'])}) from {STORE_PATH.name}?")
        if st.checkbox("Yes, delete it") and st.button("Delete latest draw"):
            delete_last()
            st.session_state["flash"] = ("info", "Latest draw deleted. Retrain in the Model tab to update the model.")
            st.rerun()
    st.caption(f"A one-time copy of the original file is kept as `{backup_path().name}` before the first change.")

# --------------------------------------------------------------------- next sets
with tab_sets:
    if bundle is None:
        st.warning("No trained model yet. Click **Retrain now** in the Model tab.")
    else:
        if n_untrained:
            st.info(f"{n_untrained} draw(s) are newer than the model's training data (up to {trained_to.date()}). "
                    "They are used as history here; retrain in the Model tab so the model learns from them too.")
        n_sets = st.number_input("Number of sets", 1, 10, 5)
        r = show_sets(bundle, draws, n_sets, key="tab")
        with st.expander("All 49 numbers ranked"):
            nums = pd.DataFrame(r["numbers"])
            nums["probability"] = (nums["probability"] * 100).round(3)
            nums["hist_freq"] = (nums["hist_freq"] * 100).round(1)
            st.dataframe(nums[["rank", "number", "probability", "score", "hist_freq", "recent_count",
                               "last_seen_date", "draws_since_last"]].rename(columns={
                "probability": "probability %", "hist_freq": "historical freq %", "recent_count": "last 20 draws"}),
                hide_index=True, use_container_width=True)
            st.caption(f"Calibrated probabilities sum to {r['prob_sum']:.2f} (six numbers are drawn), "
                       "not 100%. A fair draw gives 12.245% per number.")
        ev = r["evidence"]
        st.caption(f"Evidence on unseen test draws: {ev['verdict']} (Holm-adjusted p = {ev['p_hits10_holm']:.2f}).")
        st.warning(r["disclaimer"])

# ------------------------------------------------------------------------- model
with tab_model:
    if bundle is not None:
        m = bundle["metadata"]
        st.subheader(f"Current model: {m['best_model_label']}")
        st.markdown(f"Trained {m['created_at'].replace('T', ' ').replace('Z', ' UTC')} on "
                    f"**{m['training_data']['n_draws']} draws** ({m['training_data']['date_min']} → "
                    f"{m['training_data']['date_max']}). Chosen by {m['selection']['rule']}.")
        t, e = m["test_metrics"], m["expected_random"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg hits in top 6 (test)", f"{t['hits@6']:.3f}", f"{t['hits@6'] - e['hits@6']:+.3f} vs random")
        c2.metric("Avg hits in top 10 (test)", f"{t['hits@10']:.3f}", f"{t['hits@10'] - e['hits@10']:+.3f} vs random")
        c3.metric("Unseen test draws", m["test_period"]["n_draws"])
        st.info(f"**Evidence:** {m['test_significance']['summary']}")
        lb_path = DEFAULT_ARTIFACT_DIR / "leaderboard.csv"
        if lb_path.exists():
            lb = pd.read_csv(lb_path)
            lb = lb[lb["split"] == "test"].sort_values("hits@10", ascending=False)
            st.markdown("**All algorithms on the unseen test draws**")
            st.dataframe(lb[["label", "hits@6", "hits@10", "log_loss", "roc_auc", "p_hits10_holm"]].round(4),
                         hide_index=True, use_container_width=True)
    st.subheader("Retrain")
    st.write(f"Runs `notebook.ipynb` on all {len(draws)} draws in `{STORE_PATH.name}`"
             f"{f' ({n_untrained} not yet learned)' if n_untrained else ''}: benchmarks all algorithms, "
             "picks the best and replaces the model. About 3 minutes.")
    if st.button("Retrain now", type="primary"):
        ok, err = run_notebook()
        if ok:
            st.session_state["flash"] = ("success", "Model retrained. See the Next sets tab.")
            st.rerun()
        st.code(err)
