"""DrawSense: TOTO next-draw ranked sets (Streamlit app).

Run:  uv run streamlit run app.py      (opens http://localhost:8501)

* Add results - save a new draw to toto_results.csv -> retrain in the background -> next 5 sets
* Next sets   - ranked six-number sets from the trained model
* Model       - which algorithm was chosen, its test evidence, manual retrain

Retraining runs notebook.ipynb in a separate background process (toto_ml.retrain). The page stays
usable, a progress bar shows where it is, and a refresh or second click never starts a duplicate.
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from toto_ml import retrain
from toto_ml.artifact import DEFAULT_ARTIFACT_DIR, load_artifact, next_draw_date, predict_next
from toto_ml.store import STORE_PATH, add_draw, backup_path, delete_last, load_store, merge_upload, validate_new_draw

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


def quick_update(draws) -> bool:
    """Refit the chosen model (CatBoost) on all draws in a few seconds; no re-benchmarking."""
    from toto_ml.artifact import refit
    b = bundle_or_none()
    if b is None:
        st.error("No trained model yet. Run the full benchmark once (Model tab).")
        return False
    with st.spinner(f"Updating {b['metadata']['best_model_label']} with the new draws…"):
        refit(b, draws)
    st.cache_resource.clear()
    return True


def start_retrain() -> None:
    if retrain.start():
        st.session_state["watch_retrain"] = True
        st.session_state["seen_finish"] = None
    else:
        st.info("A retrain is already running. Its progress is shown at the top of the page.")


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


@st.fragment(run_every=2)
def retrain_panel():
    """Live status of the background retrain; refreshes itself every 2 s without reloading the page."""
    s = retrain.read_status()
    state = s.get("state", "idle")
    if state == "running":
        elapsed = retrain.fmt_elapsed(time.time() - s.get("started_at", time.time()))
        detail = f" · {s['detail']}" if s.get("detail") else ""
        st.progress(min(float(s.get("progress", 0)), 1.0), text=f"**Retraining** · {s.get('step', '')}{detail} · {elapsed}")
        st.caption("Running notebook.ipynb in the background (usually about 3 minutes). You can keep using the app; "
                   "refreshing or closing the page does not stop it.")
        st.session_state["watch_retrain"] = True
        return
    finished = s.get("finished_at")
    if st.session_state.get("watch_retrain") and finished and st.session_state.get("seen_finish") != finished:
        # a retrain this page was watching just ended: reload the whole app once to pick up the new model
        st.session_state["seen_finish"] = finished
        st.session_state["just_finished"] = s
        st.cache_resource.clear()
        st.rerun(scope="app")


st.title("DrawSense")
st.caption("Singapore TOTO · next-draw ranked sets")
retrain_panel()

draws = load_store()
bundle = bundle_or_none()
trained_to = pd.Timestamp(bundle["metadata"]["training_data"]["date_max"]) if bundle else None
n_untrained = int((pd.to_datetime(draws["draw_date"]) > trained_to).sum()) if bundle else len(draws)
running = retrain.is_running()

if "flash" in st.session_state:
    kind, msg = st.session_state.pop("flash")
    getattr(st, kind)(msg)
if st.session_state.pop("show_sets_after_update", False) and bundle is not None:
    show_sets(bundle, draws, 5, key="after_update")

done = st.session_state.pop("just_finished", None)
if done:
    if done.get("state") == "done":
        res = done.get("result") or {}
        st.success(f"Full benchmark finished in {retrain.fmt_elapsed(done.get('duration_s', 0))} on "
                   f"{res.get('n_draws')} draws (best algorithm: {res.get('best_model')}). New top 5 sets:")
        if bundle is not None:
            show_sets(bundle, draws, 5, key="after_retrain")
    else:
        st.error("Retraining failed.")
        st.code(done.get("error") or "", language=None)
        if retrain.LOG_PATH.exists():
            with st.expander("Retrain log"):
                st.code(retrain.LOG_PATH.read_text(encoding="utf-8", errors="replace")[-4000:], language=None)

tab_add, tab_sets, tab_model = st.tabs(["Add results", "Next sets", "Model"])

# ------------------------------------------------------------------- add results
with tab_add:
    last_date = pd.to_datetime(draws["draw_date"]).max().date()
    st.markdown(f"**{len(draws)} draws** in `{STORE_PATH.name}` "
                f"({pd.to_datetime(draws['draw_date']).min():%d %b %Y} → {last_date:%d %b %Y})")

    with st.form("add_draw", clear_on_submit=True):
        st.subheader("Add a new draw result")
        d = st.date_input("Draw date", value=next_draw_date(last_date), format="DD/MM/YYYY")
        main = st.multiselect("Six winning numbers (click to pick)", list(range(1, 50)), max_selections=6)
        add = st.number_input("Additional number", 1, 49, value=None, placeholder="1–49")
        auto = st.checkbox("Update the model with this draw and show the top 5 sets (a few seconds)", value=True)
        submitted = st.form_submit_button("Save draw", type="primary")
    if submitted:
        errs = validate_new_draw(draws, d, main, add) if add else ["Enter the additional number."]
        if errs:
            for e in errs:
                st.error(e)
        else:
            new_draws = add_draw(d, main, add)
            msg = f"Saved to {STORE_PATH.name}: {d:%a %d %b %Y} · {sorted(main)} + {add}."
            if auto and quick_update(new_draws):
                msg += " Model updated with it. New top 5 sets:"
                st.session_state["show_sets_after_update"] = True
            else:
                msg += " Update the model later in the Model tab."
            st.session_state["flash"] = ("success", msg)
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
            if res["added"] and quick_update(load_store()):
                st.session_state["flash"] = ("success", f"Added {res['added']} draw(s) and updated the model. "
                                                        "New top 5 sets:")
                st.session_state["show_sets_after_update"] = True
                st.rerun()
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
            quick_update(load_store())
            st.session_state["flash"] = ("info", "Latest draw deleted and the model updated.")
            st.rerun()
    st.caption(f"A one-time copy of the original file is kept as `{backup_path().name}` before the first change.")

# --------------------------------------------------------------------- next sets
with tab_sets:
    if bundle is None:
        st.warning("No trained model yet. Click **Retrain now** in the Model tab.")
    else:
        if n_untrained:
            st.info(f"{n_untrained} draw(s) are newer than the model's training data (up to {trained_to.date()}). "
                    "They are used as history here; click **Update model** in the Model tab so it learns from them.")
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
        upd = m.get("last_update")
        if upd:
            st.caption(upd["note"])
    st.subheader("Update model")
    if n_untrained:
        st.write(f"{n_untrained} draw(s) in `{STORE_PATH.name}` are not learned yet. Refit "
                 f"{bundle['metadata']['best_model_label'] if bundle else 'the model'} on all {len(draws)} draws "
                 "(same algorithm and settings, a few seconds).")
    else:
        st.write("The model is up to date with every draw. Nothing to do; just predict from it.")
    if st.button("Update model", type="primary", disabled=not n_untrained or bundle is None or running):
        if quick_update(draws):
            st.session_state["flash"] = ("success", "Model updated. New top 5 sets:")
            st.session_state["show_sets_after_update"] = True
            st.rerun()

    with st.expander("Full benchmark (occasional): re-run notebook.ipynb"):
        st.write("Re-tests all 11 algorithms and picks the best one again (about 3 minutes, in the background with "
                 "a progress bar at the top). Not needed after each draw; run it only if you want to re-check the "
                 "algorithm choice, for example after many new draws.")
        if st.button("Run full benchmark", disabled=running,
                     help="Already running; see the progress bar at the top" if running else None):
            start_retrain()
            st.rerun()
