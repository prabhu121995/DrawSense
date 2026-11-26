"""DrawSense: Singapore TOTO next move (Streamlit app).

Run:  uv run streamlit run app.py      (opens http://localhost:8501)

* Your next move    - shown first, read from artifacts/next_move.csv (saved once, not recomputed per visit)
* Get latest results - ETL: scrape new official results -> toto_results.csv -> model update -> next move
* Number guide      - all 49 numbers: model rank and prize value
* Add results       - add a draw by hand or from a file
* Your numbers in history - which past draws share numbers with a set
* Model             - the model, its validation, updates and the full benchmark
"""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from toto_ml import retrain
from toto_ml.artifact import DEFAULT_ARTIFACT_DIR, load_artifact, next_draw_date
from toto_ml.evaluation.repeats import p_any_earlier, p_share_at_least, past_matches, repeat_report
from toto_ml.nextmove import is_current, load_next_move, refresh_next_move
from toto_ml.store import STORE_PATH, add_draw, backup_path, delete_last, load_store, merge_upload, validate_new_draw

MODEL_FILE = DEFAULT_ARTIFACT_DIR / "model.joblib"

st.set_page_config(page_title="DrawSense", layout="centered")
st.markdown("""<style>
.ball{display:inline-flex;align-items:center;justify-content:center;width:42px;height:42px;border-radius:50%;
  background:#2a78d6;color:#fff;font-weight:700;font-size:17px;margin:2px 4px 2px 0;font-variant-numeric:tabular-nums}
.ball.sm{width:28px;height:28px;font-size:12px}
.ball.add{background:transparent;color:inherit;border:2px dashed #2a78d6}
.setrow{display:flex;align-items:center;gap:14px;padding:10px 0;border-bottom:1px solid rgba(128,128,128,.25);flex-wrap:wrap}
.setrank{font-size:22px;font-weight:700;width:40px;opacity:.8}
.setmeta{font-size:13px;opacity:.85;margin-left:auto;text-align:right;line-height:1.5}
.value{display:inline-block;padding:2px 8px;border-radius:999px;background:rgba(11,122,59,.14);color:#0b7a3b;font-weight:700}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;background:rgba(42,120,214,.12);color:#2a78d6;font-weight:600}
</style>""", unsafe_allow_html=True)


def balls(nums, cls=""):
    return "".join(f'<span class="ball {cls}">{int(n)}</span>' for n in nums)


@st.cache_resource(show_spinner=False)
def get_bundle(mtime: float):
    return load_artifact(DEFAULT_ARTIFACT_DIR)


def bundle_or_none():
    if not MODEL_FILE.exists():
        return None
    return get_bundle(MODEL_FILE.stat().st_mtime)  # reloads automatically after an update


def quick_update(draws) -> bool:
    """Refit the full-history model on all draws (seconds) and save the new next move."""
    from toto_ml.artifact import refit
    b = bundle_or_none()
    if b is None:
        st.error("No model yet. Run the full benchmark once (Model tab).")
        return False
    with st.spinner("Updating the model and building your next move…"):
        b = refit(b, draws)
        refresh_next_move(b, draws)
    st.cache_resource.clear()
    return True


def start_retrain() -> None:
    if retrain.start():
        st.session_state["watch_retrain"] = True
        st.session_state["seen_finish"] = None
    else:
        st.info("The full benchmark is already running. Its progress is shown at the top of the page.")


@st.fragment(run_every=2)
def retrain_panel():
    """Live status of the background benchmark; refreshes itself every 2 s without reloading the page."""
    s = retrain.read_status()
    if s.get("state") == "running":
        elapsed = retrain.fmt_elapsed(time.time() - s.get("started_at", time.time()))
        detail = f" · {s['detail']}" if s.get("detail") else ""
        st.progress(min(float(s.get("progress", 0)), 1.0), text=f"**Full benchmark** · {s.get('step', '')}{detail} · {elapsed}")
        st.caption("Running in the background (about 3 minutes). You can keep using the app.")
        st.session_state["watch_retrain"] = True
        return
    finished = s.get("finished_at")
    if st.session_state.get("watch_retrain") and finished and st.session_state.get("seen_finish") != finished:
        st.session_state["seen_finish"] = finished
        st.session_state["just_finished"] = s
        st.cache_resource.clear()
        st.rerun(scope="app")


def ensure_next_move(bundle, draws):
    """Saved next move; rebuilt once (and saved) only when the results file has changed since."""
    df, meta = load_next_move()
    if bundle is not None and not is_current(meta, draws):
        with st.spinner("Building your next move…"):
            refresh_next_move(bundle, draws)
        df, meta = load_next_move()
    return df, meta


def show_next_move(df: pd.DataFrame, meta: dict):
    b = meta["based_on"]
    st.subheader(f"Your next move · {pd.Timestamp(meta['next_draw_date']):%a %d %b %Y}")
    st.markdown(f"Built from **{b['n_draws']} draws** · latest result {pd.Timestamp(b['last_draw_date']):%d %b %Y}: "
                f"{balls(b['last_numbers'], 'sm')} + {balls([b['last_additional']], 'sm add')}", unsafe_allow_html=True)
    rows = []
    for _, s in df.iterrows():
        nums = [s[f"n{i}"] for i in range(1, 7)]
        shape = " · typical draw shape" if str(s.get("typical_shape")) == "True" else ""
        rows.append(
            f'<div class="setrow"><div class="setrank">#{int(s["rank"])}</div><div>{balls(nums)}</div>'
            f'<div class="setmeta"><span class="tag">{s["model"]}</span><br>'
            f'<span class="value">+{s["prize_value_pct"]:.0f}% prize value</span><br>'
            f'{int(s["odd"])} odd / {int(s["even"])} even · sum {int(s["sum"])}{shape}</div></div>')
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.download_button("Download your next move (CSV)", df.to_csv(index=False),
                       f"drawsense_next_move_{meta['next_draw_date']}.csv", "text/csv")
    st.caption("**Prize value** is the estimated share of a prize for this set compared with an average set: "
               "these numbers are chosen by fewer players, so a win is shared with fewer people. "
               "**Validated model** sets come from the model checked on draws it had never seen; "
               "**Full-history model** sets come from the same model trained on every draw.")


def highlights(meta: dict, n_draws: int):
    pop, shape = meta.get("popularity", {}), meta.get("shape", {})
    c1, c2, c3 = st.columns(3)
    if pop.get("g3_least") and pop.get("g3_most"):
        c1.metric("Bigger Group 3 prize", f"+{(pop['g3_least'] / pop['g3_most'] - 1) * 100:.0f}%",
                  help=f"Median Group 3 prize: ${pop['g3_least']:,.0f} when the winning numbers were the least "
                       f"popular quarter, ${pop['g3_most']:,.0f} for the most popular quarter ({pop['n_draws']} draws).")
    if shape:
        c2.metric("Typical draw shape", f"{shape['share_of_draws'] * 100:.0f}% of draws",
                  help=f"Sum {shape['sum_lo']}–{shape['sum_hi']}, {shape['odd_lo']}–{shape['odd_hi']} odd numbers, "
                       f"numbers from {shape['min_decades']}+ different tens.")
    c3.metric("Fair play verified", f"{n_draws} draws", help="Every draw in the history checked against a perfectly fair game.")


# ============================================================================ page
st.title("DrawSense")
st.caption("Singapore TOTO · your next move, built for prize value")
retrain_panel()

draws = load_store()
bundle = bundle_or_none()
running = retrain.is_running()
last_date = pd.to_datetime(draws["draw_date"]).max().date()

top_l, top_r = st.columns([3, 2], vertical_alignment="center")
top_l.markdown(f"**{len(draws)} draws** · results up to **{last_date:%a %d %b %Y}**")
etl_clicked = top_r.button("Get latest results", type="primary", use_container_width=True, disabled=running,
                           help="Downloads new official results from Singapore Pools, saves them to "
                                f"{STORE_PATH.name}, updates the model and your next move.")
if etl_clicked:
    from toto_ml.etl import run_etl
    with st.status("Getting the latest results…", expanded=True) as status:
        box = st.empty()
        shown: list[str] = []

        def on_line(line: str):
            shown.append(line)
            box.code("\n".join(shown[-12:]), language=None)

        res = run_etl(on_line=on_line)
        if res["added"]:
            status.update(label=f"Added {res['added']} new draw(s) · results up to {res['last_draw_date']}",
                          state="complete", expanded=False)
            st.session_state["flash"] = ("success", f"Added {res['added']} new draw(s) from Singapore Pools. "
                                                    "Your next move is ready below.")
        elif res["ok"]:
            status.update(label="You already have the latest results", state="complete", expanded=False)
            st.session_state["flash"] = ("info", f"You already have the latest results (up to {res['last_draw_date']}).")
        else:
            status.update(label="Could not reach Singapore Pools", state="error", expanded=True)
            st.session_state["flash"] = ("warning", "Could not reach Singapore Pools right now. Try again in a few "
                                                    "minutes; your saved results are unchanged.")
    st.cache_resource.clear()
    st.rerun()

if "flash" in st.session_state:
    kind, msg = st.session_state.pop("flash")
    getattr(st, kind)(msg)

done = st.session_state.pop("just_finished", None)
if done:
    if done.get("state") == "done":
        st.success(f"Full benchmark finished in {retrain.fmt_elapsed(done.get('duration_s', 0))}. "
                   "Your next move is updated.")
    else:
        st.error("The full benchmark stopped before finishing.")
        st.code(done.get("error") or "", language=None)

if bundle is None:
    st.info("Run the full benchmark once to create the model (Model tab).")
    nm_df, nm_meta = None, None
else:
    nm_df, nm_meta = ensure_next_move(bundle, draws)
    if nm_df is not None:
        show_next_move(nm_df, nm_meta)
        highlights(nm_meta, len(draws))

tab_guide, tab_add, tab_hist, tab_model = st.tabs(["Number guide", "Add results", "Your numbers in history", "Model"])

# ------------------------------------------------------------------ number guide
with tab_guide:
    if nm_meta and nm_meta.get("top_numbers"):
        g = pd.DataFrame(nm_meta["top_numbers"]).rename(columns={"rank": "model rank", "value_pct": "prize value %"})
        st.write("All 49 numbers in the model's order, with the prize value each number adds to a set "
                 "(positive = chosen by fewer players than average).")
        st.dataframe(g, hide_index=True, use_container_width=True, height=420)
        pop = nm_meta.get("popularity", {})
        if pop.get("cv_rho"):
            st.caption(f"Prize value is learned from {pop['n_draws']} draws of official prize data; it matches "
                       f"how crowded each draw's winners were with a correlation of {pop['cv_rho']:.2f}.")
    else:
        st.write("The number guide appears once your next move is built.")

# ------------------------------------------------------------------- add results
with tab_add:
    st.write("The quickest way is **Get latest results** at the top. You can also add a draw yourself:")
    with st.form("add_draw", clear_on_submit=True):
        d = st.date_input("Draw date", value=next_draw_date(last_date), format="DD/MM/YYYY")
        main = st.multiselect("Six winning numbers (click to pick)", list(range(1, 50)), max_selections=6)
        add = st.number_input("Additional number", 1, 49, value=None, placeholder="1–49")
        submitted = st.form_submit_button("Save draw", type="primary")
    if submitted:
        errs = validate_new_draw(draws, d, main, add) if add else ["Enter the additional number."]
        if errs:
            for e in errs:
                st.error(e)
        else:
            new_draws = add_draw(d, main, add)
            msg = f"Saved {d:%a %d %b %Y} · {sorted(main)} + {add}."
            if quick_update(new_draws):
                msg += " Your next move is updated."
            st.session_state["flash"] = ("success", msg)
            st.rerun()

    st.markdown("**Or add from a file**")
    up = st.file_uploader("CSV or XLSX with draw_date, n1..n6, additional (only new dates are added)",
                          type=["csv", "xlsx"])
    if up is not None and st.button("Add new draws from file"):
        try:
            res = merge_upload(up.getvalue(), up.name)
            msg = f"Added {res['added']} new draw(s); {res['already_present']} were already saved."
            if res["added"] and quick_update(load_store()):
                msg += " Your next move is updated."
            st.session_state["flash"] = ("success", msg)
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    st.markdown("**Latest draws**")
    show = draws.tail(10).iloc[::-1].copy()
    show["draw_date"] = pd.to_datetime(show["draw_date"]).dt.strftime("%a %d %b %Y")
    st.dataframe(show[["draw_date", "n1", "n2", "n3", "n4", "n5", "n6", "additional"]], hide_index=True,
                 use_container_width=True)
    with st.expander("Correct the latest draw"):
        lr = draws.iloc[-1]
        st.write(f"Remove the latest draw ({pd.Timestamp(lr['draw_date']):%d %b %Y}: "
                 f"{[int(lr[f'n{i}']) for i in range(1, 7)]} + {int(lr['additional'])}) so you can enter it again?")
        if st.checkbox("Yes, remove it") and st.button("Remove latest draw"):
            delete_last()
            quick_update(load_store())
            st.session_state["flash"] = ("info", "Latest draw removed and your next move updated.")
            st.rerun()
    st.caption(f"A copy of the original file is kept as `{backup_path().name}`.")

# -------------------------------------------------------- your numbers in history
@st.cache_data(show_spinner="Checking the whole draw history…")
def cached_repeat_report(draws_key: tuple) -> dict:  # key changes whenever a draw is added or removed
    return repeat_report(draws, n_sims=150)


with tab_hist:
    lr = draws.iloc[-1]
    latest = [int(lr[f"n{i}"]) for i in range(1, 7)]
    default = [int(nm_df.iloc[0][f"n{i}"]) for i in range(1, 7)] if nm_df is not None else latest
    pick = st.multiselect("Six numbers (a set from your next move, a draw result or your own ticket)",
                          list(range(1, 50)), default=default, max_selections=6)
    if len(pick) == 6:
        hits = past_matches(draws, pick, min_shared=3)
        if sorted(pick) == sorted(latest):
            hits = hits[hits["draw_number"] != lr["draw_number"]]
        st.markdown(balls(sorted(pick)), unsafe_allow_html=True)
        st.write(f"**{len(hits)}** past draws share 3 or more of these numbers.")
        if len(hits):
            show_h = hits.copy()
            show_h["draw_date"] = pd.to_datetime(show_h["draw_date"]).dt.strftime("%a %d %b %Y")
            st.dataframe(show_h, hide_index=True, use_container_width=True)
        st.caption(f"Sharing 3+ numbers with a past draw is common in a fair game: across {len(draws)} draws "
                   f"about {p_share_at_least(3) * len(draws):.1f} such matches are typical for any six numbers "
                   f"({p_any_earlier(3, len(draws)):.0%} of sets have at least one).")
    else:
        st.caption("Pick six numbers.")

    st.markdown("**Fair play check: the whole draw history**")
    rep = cached_repeat_report(tuple(draws["draw_number"]))
    st.markdown(f"Pairs of draws sharing 3+ numbers: **{rep['pairs3_real']}** in the real history, "
                f"**{rep['pairs3_expected']:.0f}** in a perfectly fair game. The history matches a fair game.")
    st.dataframe(rep["best_table"].rename(columns={"random draws (average)": "fair game (average)"}),
                 hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------- model
with tab_model:
    if bundle is not None:
        m = bundle["metadata"]
        st.subheader(m["best_model_label"])
        st.markdown(f"Trained on **{m['training_data']['n_draws']} draws** ({m['training_data']['date_min']} → "
                    f"{m['training_data']['date_max']}), updated {m['created_at'].replace('T', ' ').replace('Z', ' UTC')}.")
        t, e = m["test_metrics"], m["expected_random"]
        st.metric(f"Avg hits in top 10 · {m['test_period']['n_draws']} unseen draws", f"{t['hits@10']:.2f}",
                  f"{t['hits@10'] - e['hits@10']:+.2f} vs baseline {e['hits@10']:.2f}")
        lb_path = DEFAULT_ARTIFACT_DIR / "leaderboard.csv"
        if lb_path.exists():
            lb = pd.read_csv(lb_path)
            lb = lb[lb["split"] == "test"].sort_values("hits@10", ascending=False)
            st.markdown("**All algorithms on the unseen draws**")
            st.dataframe(lb[["label", "hits@6", "hits@10", "roc_auc"]].round(3), hide_index=True,
                         use_container_width=True)
        trained_to = pd.Timestamp(m["training_data"]["date_max"])
        n_untrained = int((pd.to_datetime(draws["draw_date"]) > trained_to).sum())
        st.markdown("**Update model**")
        if n_untrained:
            st.write(f"{n_untrained} new draw(s) are ready to learn from (a few seconds).")
        else:
            st.write("The model knows every draw.")
        if st.button("Update model", disabled=not n_untrained or running):
            if quick_update(draws):
                st.session_state["flash"] = ("success", "Model updated. Your next move is ready.")
                st.rerun()

    with st.expander("Full benchmark (occasional)"):
        st.write("Re-tests all 11 algorithms on your full history and rebuilds the model (about 3 minutes, in the "
                 "background with a progress bar at the top). Useful after many new draws.")
        if st.button("Run full benchmark", disabled=running):
            start_retrain()
            st.rerun()
