"""DrawSense: Singapore TOTO + 4D (Streamlit app, mobile first).

Run:  uv run streamlit run app.py      (opens http://localhost:8501)

* Get latest results - ETL for both games: scrape TOTO + 4D -> CSV files -> drawsense.db -> TOTO model + next move
* Game switch        - TOTO or 4D (a filter over the same database; ?game=4d in the URL opens 4D)
* TOTO               - your next move (model), number guide, Axis & Mirror, add results, history, model
* 4D                 - any draw with Axis & Mirror metrics, number check, history filters, stats, the chart
* Database           - what is stored in drawsense.db, ETL log, download
"""
from __future__ import annotations

import time
from datetime import timedelta

import pandas as pd
import streamlit as st

from toto_ml import db, fourd_inspired, retrain
from toto_ml.artifact import DEFAULT_ARTIFACT_DIR, load_artifact, next_draw_date
from toto_ml.evaluation.repeats import p_any_earlier, p_share_at_least, past_matches, repeat_report
from toto_ml.metrics import metrics
from toto_ml.nextmove import is_current, load_next_move, refresh_next_move
from toto_ml.store import STORE_PATH, add_draw, backup_path, delete_last, load_store, merge_upload, validate_new_draw

MODEL_FILE = DEFAULT_ARTIFACT_DIR / "model.joblib"
FOURD_DAYS = {2, 5, 6}  # 4D draws: Wednesday, Saturday, Sunday
PRIZE_LABEL = {"1st": "1st Prize", "2nd": "2nd Prize", "3rd": "3rd Prize", "starter": "Starter",
               "consolation": "Consolation", "main": "Winning", "additional": "Additional"}

st.set_page_config(page_title="DrawSense", page_icon="🎯", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
/* ---- mobile first: use the whole screen ---- */
[data-testid="stHeader"],[data-testid="stDecoration"],footer,#MainMenu{display:none!important}
.block-container,[data-testid="stMainBlockContainer"]{padding:.6rem .7rem 4rem!important;max-width:100%!important}
@media(min-width:768px){.block-container,[data-testid="stMainBlockContainer"]{padding:1rem 1.6rem 4rem!important}}
@media(min-width:1400px){.block-container,[data-testid="stMainBlockContainer"]{padding:1.2rem 3rem 4rem!important}}
h1,h2,h3{margin-top:.2rem!important}
.stTabs [data-baseweb="tab-list"]{gap:0;overflow-x:auto;scrollbar-width:none;flex-wrap:nowrap}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar{display:none}
.stTabs [data-baseweb="tab"]{padding:8px 12px;white-space:nowrap;font-size:15px}
[data-testid="stMetric"]{background:rgba(128,128,128,.07);border-radius:12px;padding:10px 12px}
/* ---- brand bar ---- */
.brand{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:2px}
.brand b{font-size:clamp(22px,6vw,30px);letter-spacing:-.01em}
.brand span{opacity:.7;font-size:14px}
.status{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:6px 0 4px}
.status div{background:rgba(128,128,128,.08);border-radius:12px;padding:8px 10px;font-size:13px;line-height:1.35}
.status b{font-size:15px}
/* ---- balls and set cards ---- */
.ball{display:inline-flex;align-items:center;justify-content:center;width:clamp(34px,10vw,44px);
  height:clamp(34px,10vw,44px);border-radius:50%;background:#2a78d6;color:#fff;font-weight:700;
  font-size:clamp(14px,4.2vw,17px);margin:2px 3px 2px 0;font-variant-numeric:tabular-nums}
.ball.sm{width:28px;height:28px;font-size:12px}
.ball.add{background:transparent;color:inherit;border:2px dashed #2a78d6}
.sets{display:grid;grid-template-columns:1fr;gap:10px;margin:6px 0 10px}
@media(min-width:1000px){.sets{grid-template-columns:1fr 1fr}}
.card{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:12px 12px 10px}
.cardhead{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap}
.rank{font-size:20px;font-weight:800;opacity:.8}
.setmeta{font-size:13px;opacity:.9;margin-top:6px;line-height:1.6}
.value{display:inline-block;padding:2px 8px;border-radius:999px;background:rgba(31,157,85,.16);color:#1f9d55;font-weight:700}
.tag{display:inline-block;padding:2px 8px;border-radius:999px;background:rgba(42,120,214,.12);color:#2a78d6;font-weight:600}
/* ---- metrics chips ---- */
.ram{display:flex;gap:4px;flex-wrap:wrap;justify-content:center;font-size:11.5px;margin-top:3px}
.ram i{font-style:normal;padding:1px 6px;border-radius:999px;background:rgba(128,128,128,.13);font-variant-numeric:tabular-nums}
.ram i.r{background:rgba(180,110,0,.15)} .ram i.a{background:rgba(200,40,40,.13)} .ram i.m{background:rgba(42,120,214,.14)}
.legend{font-size:12px;opacity:.8;margin:4px 0 8px}
/* ---- 4D draw ---- */
.fd-top{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.fd-tile{border-radius:12px;padding:10px 4px 8px;text-align:center;background:rgba(232,120,20,.10)}
.fd-tile .lbl{font-size:12px;font-weight:700;opacity:.75;text-transform:uppercase;letter-spacing:.05em}
.fd-num{font-size:clamp(24px,7.5vw,38px);font-weight:800;letter-spacing:.06em;font-variant-numeric:tabular-nums;line-height:1.15}
.sec{font-size:13px;font-weight:700;opacity:.75;margin:12px 0 6px;text-transform:uppercase;letter-spacing:.05em}
.fd-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}
@media(min-width:560px){.fd-grid{grid-template-columns:repeat(5,1fr)}}
.fd-cell{border-radius:10px;padding:6px 2px;text-align:center;background:rgba(128,128,128,.07)}
.fd-cell .n{font-size:clamp(18px,5.5vw,22px);font-weight:700;letter-spacing:.05em;font-variant-numeric:tabular-nums}
.tm{display:grid;grid-template-columns:repeat(auto-fill,minmax(64px,1fr));gap:6px}
.tm div{text-align:center}
.big{font-size:clamp(30px,10vw,44px);font-weight:800;letter-spacing:.06em;font-variant-numeric:tabular-nums}
</style>""", unsafe_allow_html=True)


# ============================================================================ helpers
def balls(nums, cls=""):
    return "".join(f'<span class="ball {cls}">{int(n)}</span>' for n in nums)


def ram(number, width) -> str:
    m = metrics(number, width)
    return (f'<div class="ram"><i class="r">R {m["root"]}</i><i class="a">A {m["axis"]}</i>'
            f'<i class="m">M {m["mirror"]}</i></div>')


LEGEND = ('<div class="legend"><b>R</b> root (digit sum to one digit) · <b>A</b> axis (0↔5 1↔6 2↔7 3↔8 4↔9) · '
          '<b>M</b> mirror (digits reversed)</div>')


def nice(d) -> str:
    return pd.Timestamp(d).strftime("%a %d %b %Y")


def next_fourd_date(last) -> pd.Timestamp:
    d = pd.Timestamp(last) + timedelta(days=1)
    while d.weekday() not in FOURD_DAYS:
        d += timedelta(days=1)
    return d


def db_key() -> float:
    """Changes whenever the database file is rewritten, so cached queries refresh."""
    return db.DB_PATH.stat().st_mtime if db.DB_PATH.exists() else 0.0


@st.cache_data(show_spinner=False)
def q_summary(key):
    return db.summary()


@st.cache_data(show_spinner=False)
def q_fourd(key):
    return db.fourd_draws()


@st.cache_data(show_spinner=False)
def q_numbers(key, game):
    return db.draw_numbers(game)


@st.cache_data(show_spinner=False)
def q_chart(key):
    return db.chart()


@st.cache_resource(show_spinner=False)
def get_bundle(mtime: float):
    return load_artifact(DEFAULT_ARTIFACT_DIR)


def bundle_or_none():
    if not MODEL_FILE.exists():
        return None
    return get_bundle(MODEL_FILE.stat().st_mtime)  # reloads automatically after an update


def toto_changed(draws) -> bool:
    """After a TOTO change made in the app: refresh the database, then refit the model (seconds)."""
    db.sync_toto()
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


def number_check(game: str, number: str):
    """Metrics of a number plus every past draw of the number, its axis, its mirror (and 4D permutations)."""
    width = db.GAMES[game][1]
    res = db.lookup(game, number)
    m = res["metrics"]
    st.markdown(f'<div class="card" style="text-align:center"><div class="big">{m["number"]}</div>'
                f'{ram(m["number"], width)}</div>', unsafe_allow_html=True)
    parts = [("Same number", "number"), (f"Axis {m['axis']}", "axis"), (f"Mirror {m['mirror']}", "mirror")]
    if game == "4d":
        parts.append(("Any order (permutation)", "permutation"))
    cols = st.columns(len(parts))
    for c, (label, key) in zip(cols, parts):
        df = res[key]
        last = f"last {pd.Timestamp(df['draw_date'].iloc[0]):%d %b %y}" if len(df) else "never drawn"
        c.metric(label, f"{len(df)}×", last, delta_color="off", delta_arrow="off")
    for label, key in parts:
        df = res[key]
        if len(df):
            with st.expander(f"{label}: {len(df)} time(s)"):
                show = df.assign(prize=df["prize"].map(PRIZE_LABEL).fillna(df["prize"]))
                st.dataframe(show, hide_index=True, width="stretch")


def chart_tab():
    st.markdown("**Axis & Mirror Numbers Reference Chart (01 – 89)**")
    st.markdown(LEGEND, unsafe_allow_html=True)
    ch = q_chart(db_key())
    find = st.text_input("Find a number in the chart", max_chars=2, placeholder="e.g. 27", key="chart_find")
    if find.strip().isdigit():
        ch = ch[(ch["number"] == find.zfill(2)) | (ch["axis"] == find.zfill(2)) | (ch["mirror"] == find.zfill(2))]
    st.dataframe(ch, hide_index=True, width="stretch", height=420)
    st.caption("Stored in the database table `number_metrics` (00–99; `in_chart` marks 01–89 from the document). "
               "The same rules are applied to every drawn number in `draw_numbers`.")


# ============================================================================ header + ETL
db.ensure_db()
retrain_panel()
running = retrain.is_running()
summ = q_summary(db_key()).set_index("game")


def game_status(g):
    r = summ.loc[g]
    if not r["draws"]:
        return f"<div><b>{r['name']}</b><br>no draws yet</div>"
    return f"<div><b>{r['name']}</b> · {int(r['draws'])} draws<br>latest {nice(r['last_date'])}</div>"


st.markdown('<div class="brand"><b>DrawSense</b><span>Singapore TOTO &amp; 4D · one database</span></div>'
            f'<div class="status">{game_status("toto")}{game_status("4d")}</div>', unsafe_allow_html=True)

if st.button("⟳  Get latest results (TOTO + 4D)", type="primary", width="stretch", disabled=running,
             help="Downloads new official TOTO and 4D results from Singapore Pools, stores both in drawsense.db, "
                  "updates the TOTO model and your next move."):
    from toto_ml.etl import run_etl
    with st.status("Getting the latest results…", expanded=True) as status:
        box = st.empty()
        shown: list[str] = []

        def on_line(line: str):
            shown.append(line)
            box.code("\n".join(shown[-10:]), language=None)

        res = run_etl(on_line=on_line)
        f = res["fourd"]
        msg = f"TOTO +{res['added']} (to {res['last_draw_date']}) · 4D +{f['added']} (to {f['last_draw_date']})"
        if res["ok"] and f["ok"]:
            status.update(label=msg, state="complete", expanded=False)
            kind = "success" if res["added"] or f["added"] else "info"
            text = (f"Saved to the database: {msg}." if res["added"] or f["added"]
                    else "You already have the latest TOTO and 4D results.")
        else:
            status.update(label="Could not reach Singapore Pools for every game", state="error", expanded=True)
            kind, text = "warning", (f"{msg}. Part of the download failed; try again in a few minutes. "
                                     "Your saved results are unchanged.")
        st.session_state["flash"] = (kind, text)
    st.cache_resource.clear()
    st.cache_data.clear()
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

qp = st.query_params.get("game", "toto").lower()
game_label = st.segmented_control("Game", ["TOTO", "4D"], default="4D" if qp == "4d" else "TOTO",
                                  required=True, width="stretch", label_visibility="collapsed", key="game")
game = "4d" if game_label == "4D" else "toto"
if st.query_params.get("game") != game:
    st.query_params["game"] = game


# ============================================================================ TOTO
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
    st.subheader(f"Your next move · {nice(meta['next_draw_date'])}")
    st.markdown(f"Built from **{b['n_draws']} draws** · latest {pd.Timestamp(b['last_draw_date']):%d %b %Y}: "
                f"{balls(b['last_numbers'], 'sm')} + {balls([b['last_additional']], 'sm add')}", unsafe_allow_html=True)
    cards = []
    for _, s in df.iterrows():
        nums = [s[f"n{i}"] for i in range(1, 7)]
        shape = " · typical shape" if str(s.get("typical_shape")) == "True" else ""
        cards.append(
            f'<div class="card"><div class="cardhead"><span class="rank">#{int(s["rank"])}</span>'
            f'<span class="tag">{s["model"]}</span></div><div>{balls(nums)}</div>'
            f'<div class="setmeta"><span class="value">+{s["prize_value_pct"]:.0f}% prize value</span> '
            f'{int(s["odd"])} odd / {int(s["even"])} even · sum {int(s["sum"])}{shape}</div></div>')
    st.markdown(f'<div class="sets">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.download_button("Download your next move (CSV)", df.to_csv(index=False),
                       f"drawsense_next_move_{meta['next_draw_date']}.csv", "text/csv", width="stretch")
    st.caption("**Prize value** is the estimated share of a prize for this set compared with an average set: "
               "these numbers are chosen by fewer players, so a win is shared with fewer people. "
               "**Validated model** sets come from the model checked on draws it had never seen; "
               "**Full-history model** sets come from the same model trained on every draw.")


@st.cache_data(show_spinner="Reading the last three 4D draws…")
def cached_fourd_inspired(draws_key: tuple, fourd_mtime: float, _draws) -> dict:  # keys change on new draws
    return fourd_inspired.predict(_draws)


def show_fourd_inspired(draws: pd.DataFrame):
    """Third source: sets from the last three 4D draws. Separate from the two models above."""
    path = db.FOURD_PATH
    r = cached_fourd_inspired(tuple(draws["draw_number"]), path.stat().st_mtime if path.exists() else 0.0, draws)
    st.subheader("4D-inspired sets")
    if not r.get("ok"):
        st.info(r.get("reason", "No 4D results yet. Tap **Get latest results** first."))
        return
    src = " · ".join(f"{nice(x['draw_date'])} ({x['first']}, {x['second']}, {x['third']})" for x in r["based_on_4d"])
    st.markdown(f"From the last three 4D draws: {src}")
    cards = [f'<div class="card"><div class="cardhead"><span class="rank">#{s["rank"]}</span>'
             f'<span class="tag">{fourd_inspired.LABEL}</span></div><div>{balls(s["numbers"])}</div>'
             f'<div class="setmeta">{s["odd"]} odd / {s["even"]} even · sum {s["sum"]}</div></div>'
             for s in r["sets"]]
    st.markdown(f'<div class="sets">{"".join(cards)}</div>', unsafe_allow_html=True)

    b = r["backtest"]
    if b["n_draws"]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Set #1 hits per draw", f"{b['set1_hits_avg']:.2f}",
                  f"{b['set1_hits_avg'] - b['set_hits_expected']:+.2f} vs random", delta_color="off",
                  help=f"Average winning numbers in set #1 over {b['n_draws']} past draws. "
                       f"Six random numbers average {b['set_hits_expected']:.2f}.")
        c2.metric("Pool of 18: hits per draw", f"{b['pool_hits_avg']:.2f}",
                  f"{b['pool_hits_avg'] - b['pool_hits_expected']:+.2f} vs random", delta_color="off",
                  help=f"18 random numbers average {b['pool_hits_expected']:.2f} of the 6 winning numbers.")
        c3.metric("Pool had 3+ winners", f"{b['pool_3plus'] * 100:.0f}% of draws",
                  f"random: {b['pool_3plus_expected'] * 100:.0f}%", delta_color="off",
                  help="Share of past draws where the 18-number pool held at least 3 winning numbers.")
        st.caption(f"Checked on **{b['n_draws']} past TOTO draws**, each using only the three 4D draws before it. "
                   "Compare each figure with what random numbers score: a method that really uses 4D to "
                   "predict TOTO would beat random by a clear margin.")
    with st.expander("How the 4D-inspired numbers are found"):
        st.markdown(
            "1. Every 4D number from the last three draws (1st, 2nd, 3rd, 10 Starter, 10 Consolation) is split "
            "into front and back pairs: 8064 → 80, 64.\n"
            "2. Each pair counts as itself, as its **axis** number (0↔5 1↔6 2↔7 3↔8 4↔9: 64 → 19) and as its "
            "**mirror** (reversed: 64 → 46). Only 1–49 is kept.\n"
            "3. Weights: the pair itself 1, axis or mirror ½; 1st/2nd/3rd prize ×2; newest 4D draw ×1, then ×0.85 "
            "and ×0.7.\n"
            "4. The 18 heaviest numbers are the pool; sets are ranks 1–6, 7–12 and 13–18.")
        st.dataframe(pd.DataFrame(r["pool"]), hide_index=True, width="stretch")
        if b["n_draws"]:
            st.markdown("**Every past draw checked**")
            st.dataframe(b["table"].iloc[::-1], hide_index=True, width="stretch", height=300)


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


@st.cache_data(show_spinner="Checking the whole draw history…")
def cached_repeat_report(draws_key: tuple, _draws) -> dict:  # key changes whenever a draw is added or removed
    return repeat_report(_draws, n_sims=150)


def toto_page():
    draws = load_store()
    bundle = bundle_or_none()
    last_date = pd.to_datetime(draws["draw_date"]).max().date()

    if bundle is None:
        st.info("Run the full benchmark once to create the model (Model tab).")
        nm_df, nm_meta = None, None
    else:
        nm_df, nm_meta = ensure_next_move(bundle, draws)
        if nm_df is not None:
            show_next_move(nm_df, nm_meta)
            highlights(nm_meta, len(draws))
    show_fourd_inspired(draws)

    t_guide, t_am, t_add, t_hist, t_model = st.tabs(["Number guide", "Axis & Mirror", "Add results", "History", "Model"])

    with t_guide:
        if nm_meta and nm_meta.get("top_numbers"):
            g = pd.DataFrame(nm_meta["top_numbers"]).rename(columns={"rank": "model rank", "value_pct": "prize value %"})
            st.write("All 49 numbers in the model's order, with the prize value each number adds to a set "
                     "(positive = chosen by fewer players than average).")
            st.dataframe(g, hide_index=True, width="stretch", height=420)
            pop = nm_meta.get("popularity", {})
            if pop.get("cv_rho"):
                st.caption(f"Prize value is learned from {pop['n_draws']} draws of official prize data; it matches "
                           f"how crowded each draw's winners were with a correlation of {pop['cv_rho']:.2f}.")
        else:
            st.write("The number guide appears once your next move is built.")

    with t_am:
        toto_axis_mirror(draws)

    with t_add:
        st.write("The quickest way is **Get latest results** at the top. You can also add a draw yourself:")
        with st.form("add_draw", clear_on_submit=True):
            d = st.date_input("Draw date", value=next_draw_date(last_date), format="DD/MM/YYYY")
            main = st.multiselect("Six winning numbers (tap to pick)", list(range(1, 50)), max_selections=6)
            add = st.number_input("Additional number", 1, 49, value=None, placeholder="1–49")
            submitted = st.form_submit_button("Save draw", type="primary", width="stretch")
        if submitted:
            errs = validate_new_draw(draws, d, main, add) if add else ["Enter the additional number."]
            if errs:
                for e in errs:
                    st.error(e)
            else:
                new_draws = add_draw(d, main, add)
                msg = f"Saved {d:%a %d %b %Y} · {sorted(main)} + {add}."
                if toto_changed(new_draws):
                    msg += " Your next move is updated."
                st.session_state["flash"] = ("success", msg)
                st.cache_data.clear()
                st.rerun()

        st.markdown("**Or add from a file**")
        up = st.file_uploader("CSV or XLSX with draw_date, n1..n6, additional (only new dates are added)",
                              type=["csv", "xlsx"])
        if up is not None and st.button("Add new draws from file", width="stretch"):
            try:
                res = merge_upload(up.getvalue(), up.name)
                msg = f"Added {res['added']} new draw(s); {res['already_present']} were already saved."
                if res["added"] and toto_changed(load_store()):
                    msg += " Your next move is updated."
                st.session_state["flash"] = ("success", msg)
                st.cache_data.clear()
                st.rerun()
            except ValueError as e:
                st.error(str(e))

        st.markdown("**Latest draws**")
        show = draws.tail(10).iloc[::-1].copy()
        show["draw_date"] = pd.to_datetime(show["draw_date"]).dt.strftime("%a %d %b %Y")
        st.dataframe(show[["draw_date", "n1", "n2", "n3", "n4", "n5", "n6", "additional"]], hide_index=True,
                     width="stretch")
        with st.expander("Correct the latest draw"):
            lr = draws.iloc[-1]
            st.write(f"Remove the latest draw ({pd.Timestamp(lr['draw_date']):%d %b %Y}: "
                     f"{[int(lr[f'n{i}']) for i in range(1, 7)]} + {int(lr['additional'])}) so you can enter it again?")
            if st.checkbox("Yes, remove it") and st.button("Remove latest draw"):
                delete_last()
                toto_changed(load_store())
                st.session_state["flash"] = ("info", "Latest draw removed and your next move updated.")
                st.cache_data.clear()
                st.rerun()
        st.caption(f"A copy of the original file is kept as `{backup_path().name}`.")

    with t_hist:
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
                st.dataframe(show_h, hide_index=True, width="stretch")
            st.caption(f"Sharing 3+ numbers with a past draw is common in a fair game: across {len(draws)} draws "
                       f"about {p_share_at_least(3) * len(draws):.1f} such matches are typical for any six numbers "
                       f"({p_any_earlier(3, len(draws)):.0%} of sets have at least one).")
        else:
            st.caption("Pick six numbers.")

        st.markdown("**Fair play check: the whole draw history**")
        rep = cached_repeat_report(tuple(draws["draw_number"]), draws)
        st.markdown(f"Pairs of draws sharing 3+ numbers: **{rep['pairs3_real']}** in the real history, "
                    f"**{rep['pairs3_expected']:.0f}** in a perfectly fair game. The history matches a fair game.")
        st.dataframe(rep["best_table"].rename(columns={"random draws (average)": "fair game (average)"}),
                     hide_index=True, width="stretch")

    with t_model:
        toto_model(draws, bundle)


def toto_axis_mirror(draws):
    nums = q_numbers(db_key(), "toto")
    if nums.empty:
        st.info("No TOTO data in the database yet.")
        return
    last_d = nums["draw_date"].max()
    last = nums[nums["draw_date"] == last_d].sort_values(["prize", "position"], ascending=[False, True])
    st.markdown(f"**Latest draw · {nice(last_d)}**")
    st.markdown(LEGEND, unsafe_allow_html=True)
    cells = "".join(f'<div><span class="ball {"add" if r.prize == "additional" else ""}">{int(r.number)}</span>'
                    f'{ram(r.number, 2)}</div>' for r in last.itertuples())
    st.markdown(f'<div class="card"><div class="tm">{cells}</div></div>', unsafe_allow_html=True)

    st.markdown("**Number check**")
    n = st.selectbox("Pick a number (1–49)", list(range(1, 50)), index=26, key="toto_check")
    number_check("toto", f"{n:02d}")

    st.markdown("**All TOTO numbers with metrics**")
    c1, c2 = st.columns(2)
    prizes = c1.pills("Numbers", ["main", "additional"], default=["main", "additional"], selection_mode="multi",
                      format_func=PRIZE_LABEL.get, key="toto_prz")
    roots = c2.pills("Root", list(range(1, 10)), selection_mode="multi", key="toto_root")
    view = nums[nums["prize"].isin(prizes or ["main", "additional"])]
    if roots:
        view = view[view["root"].isin(roots)]
    cols = ["draw_date", "draw_number", "prize", "number", "digit_sum", "root", "axis", "mirror"]
    st.dataframe(view[cols], hide_index=True, width="stretch", height=380)
    st.download_button("Download (CSV)", view[cols].to_csv(index=False), "toto_axis_mirror.csv", "text/csv",
                       width="stretch")
    with st.expander("Axis & Mirror reference chart (01–89)"):
        chart_tab()


def toto_model(draws, bundle):
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
            st.dataframe(lb[["label", "hits@6", "hits@10", "roc_auc"]].round(3), hide_index=True, width="stretch")
        trained_to = pd.Timestamp(m["training_data"]["date_max"])
        n_untrained = int((pd.to_datetime(draws["draw_date"]) > trained_to).sum())
        st.markdown("**Update model**")
        st.write(f"{n_untrained} new draw(s) are ready to learn from (a few seconds)." if n_untrained
                 else "The model knows every draw.")
        if st.button("Update model", disabled=not n_untrained or running, width="stretch"):
            if toto_changed(draws):
                st.session_state["flash"] = ("success", "Model updated. Your next move is ready.")
                st.rerun()
    st.caption("The prediction model is built for TOTO only (6 numbers from 1–49). 4D results are stored in the "
               "same database and analysed on the 4D page.")
    with st.expander("Full benchmark (occasional)"):
        st.write("Re-tests all 11 algorithms on your full history and rebuilds the model (about 3 minutes, in the "
                 "background with a progress bar at the top). Useful after many new draws.")
        if st.button("Run full benchmark", disabled=running):
            start_retrain()
            st.rerun()


# ============================================================================ 4D
def fourd_draw_card(r: pd.Series):
    def tile(label, num):
        return f'<div class="fd-tile"><div class="lbl">{label}</div><div class="fd-num">{num}</div>{ram(num, 4)}</div>'

    def cells(prefix):
        return "".join(f'<div class="fd-cell"><div class="n">{r[f"{prefix}_{i}"]}</div>{ram(r[f"{prefix}_{i}"], 4)}</div>'
                       for i in range(1, 11) if r.get(f"{prefix}_{i}"))

    st.markdown(
        f'<div class="card"><div class="cardhead"><b>{nice(r["draw_date"])}</b>'
        f'<span class="tag">Draw {int(r["draw_number"])}</span></div>'
        f'<div class="fd-top">{tile("1st", r["first"])}{tile("2nd", r["second"])}{tile("3rd", r["third"])}</div>'
        f'<div class="sec">Starter prizes</div><div class="fd-grid">{cells("starter")}</div>'
        f'<div class="sec">Consolation prizes</div><div class="fd-grid">{cells("consolation")}</div></div>',
        unsafe_allow_html=True)


def follow_ups(sv: pd.DataFrame, key: str) -> int:
    """How often the axis/mirror of a drawn number was itself drawn in a later draw."""
    base = sv if key == "axis" else sv[sv["number"] != sv["mirror"]]
    later = sv[["draw_id", "number"]].rename(columns={"draw_id": "later", "number": key})
    hits = base[["draw_id", key]].merge(later, on=key)
    return int((hits["later"] > hits["draw_id"]).sum())


def fourd_page():
    draws = q_fourd(db_key())
    if draws.empty:
        st.info("No 4D results yet. Tap **Get latest results** to download them from Singapore Pools "
                "(the first download takes about 10 minutes, later ones only fetch new draws).")
        return
    st.caption(f"{len(draws)} draws · {nice(draws['draw_date'].iloc[-1])} → {nice(draws['draw_date'].iloc[0])} · "
               f"next draw {nice(next_fourd_date(draws['draw_date'].iloc[0]))}")
    labels = [f"{nice(d)} · #{int(n)}" for d, n in zip(draws["draw_date"], draws["draw_number"])]
    i = st.selectbox("Draw", range(len(draws)), format_func=lambda k: labels[k], key="fd_pick",
                     label_visibility="collapsed")
    st.markdown(LEGEND, unsafe_allow_html=True)
    fourd_draw_card(draws.iloc[i])

    t_check, t_hist, t_stats, t_chart = st.tabs(["Number check", "History", "Stats", "Axis & Mirror chart"])
    nums = q_numbers(db_key(), "4d")

    with t_check:
        n = st.text_input("Your 4D number", value=draws.iloc[i]["first"], max_chars=4, key="fd_check",
                          placeholder="e.g. 0725")
        if n.strip().isdigit():
            number_check("4d", n.strip().zfill(4))
        else:
            st.caption("Enter up to four digits.")

    with t_hist:
        c1, c2 = st.columns([3, 2])
        prizes = c1.pills("Prize", db.FOURD_PRIZES, default=db.FOURD_PRIZES, selection_mode="multi",
                          format_func=PRIZE_LABEL.get, key="fd_prz")
        dmin, dmax = pd.Timestamp(nums["draw_date"].min()).date(), pd.Timestamp(nums["draw_date"].max()).date()
        rng = c2.date_input("Period", (max(dmin, dmax - timedelta(days=90)), dmax), min_value=dmin, max_value=dmax,
                            format="DD/MM/YYYY", key="fd_rng")
        c3, c4 = st.columns([3, 2])
        roots = c3.pills("Root", list(range(0, 10)), selection_mode="multi", key="fd_root")
        has = c4.text_input("Number contains", max_chars=4, placeholder="e.g. 25", key="fd_has")
        view = nums[nums["prize"].isin(prizes or db.FOURD_PRIZES)]
        if isinstance(rng, (tuple, list)) and len(rng) == 2:
            view = view[(view["draw_date"] >= str(rng[0])) & (view["draw_date"] <= str(rng[1]))]
        if roots:
            view = view[view["root"].isin(roots)]
        if has.strip().isdigit():
            view = view[view["number"].str.contains(has.strip(), regex=False)]
        cols = ["draw_date", "draw_number", "prize", "number", "root", "axis", "mirror", "front", "back",
                "front_axis", "front_mirror", "back_axis", "back_mirror"]
        st.write(f"**{len(view):,}** numbers from **{view['draw_id'].nunique()}** draws")
        st.dataframe(view[cols].assign(prize=view["prize"].map(PRIZE_LABEL)), hide_index=True, width="stretch",
                     height=420)
        d1, d2 = st.columns(2)
        d1.download_button("Download filtered (CSV)", view[cols].to_csv(index=False), "4d_axis_mirror.csv",
                           "text/csv", width="stretch")
        d2.download_button("Download all draws (CSV)", draws.drop(columns=["draw_id"]).to_csv(index=False),
                           "4d_results.csv", "text/csv", width="stretch")

    with t_stats:
        prz = st.pills("Prizes counted", db.FOURD_PRIZES, default=["1st", "2nd", "3rd"], selection_mode="multi",
                       format_func=PRIZE_LABEL.get, key="fd_stat_prz")
        sv = nums[nums["prize"].isin(prz or db.FOURD_PRIZES)]
        st.markdown(f"**Root of winning numbers** · {len(sv):,} numbers")
        rc = sv["root"].value_counts().reindex(range(10), fill_value=0)
        st.bar_chart(pd.DataFrame({"numbers": rc.values}, index=[str(k) for k in rc.index]), height=220,
                     x_label="root", y_label="numbers")
        st.caption("Roots 1–9 are equally common among all 10,000 4D numbers (1,111 each, root 0 only for 0000), "
                   "so in a fair draw the bars should be roughly level.")
        a, b = st.columns(2)
        with a:
            st.markdown("**Numbers drawn more than once**")
            rep = sv.groupby("number").agg(times=("number", "size"), last=("draw_date", "max"))
            rep = rep[rep["times"] > 1].sort_values(["times", "last"], ascending=False).reset_index()
            st.dataframe(rep, hide_index=True, width="stretch", height=300)
        with b:
            st.markdown("**Most common back pairs**")
            bp = sv.groupby("back").size().sort_values(ascending=False).head(20).rename("times").reset_index()
            st.dataframe(bp, hide_index=True, width="stretch", height=300)
        st.caption(f"Follow-ups in the selected prizes: a number's **axis** won in a later draw "
                   f"{follow_ups(sv, 'axis')} time(s), its **mirror** {follow_ups(sv, 'mirror')} time(s) "
                   "(palindromes like 1221 left out of the mirror count).")

    with t_chart:
        chart_tab()


# ============================================================================ database panel
def database_panel():
    with st.expander("🗄  Database · drawsense.db"):
        st.write(f"`{db.DB_PATH}` holds both games. Filter any table with `game = 'toto'` or `game = '4d'`.")
        st.dataframe(q_summary(db_key()), hide_index=True, width="stretch")
        a, b = st.columns(2)
        a.dataframe(db.table_counts(), hide_index=True, width="stretch")
        hist = db.etl_history(10)
        b.dataframe(hist if len(hist) else pd.DataFrame({"ETL runs": ["none yet"]}), hide_index=True, width="stretch")
        if db.DB_PATH.exists():
            st.download_button("Download drawsense.db", db.DB_PATH.read_bytes(), "drawsense.db",
                               "application/vnd.sqlite3", width="stretch")
        if st.button("Rebuild database from the CSV files", width="stretch"):
            st.session_state["flash"] = ("success", f"Database rebuilt: {db.sync_all()}")
            st.cache_data.clear()
            st.rerun()


if game == "toto":
    toto_page()
else:
    fourd_page()
database_panel()
