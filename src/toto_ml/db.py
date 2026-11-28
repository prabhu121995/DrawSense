"""One SQLite database for both games: ``drawsense.db``.

    toto_results.csv  ─┐
                       ├─ sync_all() ─> drawsense.db
    fourd_results.csv ─┘

The CSV files stay the source of truth (the TOTO model trains from toto_results.csv, the scrapers append to
them). Every ETL run, and every change made in the app, rebuilds the game's rows in the database from its CSV
in one transaction, so the database always matches the files.

Tables (filter any of them with ``game = 'toto'`` or ``game = '4d'``):

    games            toto / 4d
    draws            one row per draw of either game: game, draw_number, draw_date
    toto_results     the TOTO CSV, typed (numbers, additional, Group 1-7 prize data)
    fourd_results    the 4D CSV (1st, 2nd, 3rd, 10 starter, 10 consolation; 4-digit text)
    draw_numbers     every drawn number of both games, one row each, with its Axis & Mirror metrics
                     (digit_sum, root, axis, mirror; 4D also front/back pairs)
    number_metrics   the Axis & Mirror reference chart (00-99, in_chart = the document's 01-89)
    etl_runs         log of each ETL run per game
    v_draw_number_metrics  draw_numbers + chart metrics of the front and back pair (4D)

The TOTO prediction algorithm only reads TOTO data; 4D is stored and analysed separately.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT
from .metrics import metrics, reference_chart

DB_PATH = Path(os.environ.get("TOTO_ML_DB", PROJECT_ROOT / "drawsense.db"))
FOURD_PATH = Path(os.environ.get("TOTO_ML_FOURD", PROJECT_ROOT / "fourd_results.csv"))

GAMES = {
    "toto": ("TOTO", 2, "https://www.singaporepools.com.sg/en/product/pages/toto_results.aspx"),
    "4d": ("4D", 4, "https://www.singaporepools.com.sg/en/product/pages/4d_results.aspx"),
}
TOTO_PRIZE_COLS = ["group1_prize"] + [f"g{g}_{k}" for g in range(1, 8) for k in ("share_amount", "winning_shares")]
FOURD_STARTER = [f"starter_{i}" for i in range(1, 11)]
FOURD_CONSOLATION = [f"consolation_{i}" for i in range(1, 11)]
FOURD_COLS = ["first", "second", "third"] + FOURD_STARTER + FOURD_CONSOLATION
FOURD_PRIZES = ["1st", "2nd", "3rd", "starter", "consolation"]

SCHEMA = f"""
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS games (
    game TEXT PRIMARY KEY, name TEXT NOT NULL, digits INTEGER NOT NULL, source_url TEXT);
CREATE TABLE IF NOT EXISTS draws (
    draw_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game TEXT NOT NULL REFERENCES games(game),
    draw_number INTEGER,                 -- NULL for a TOTO draw typed in by hand
    draw_date TEXT NOT NULL,             -- ISO yyyy-mm-dd
    weekday TEXT NOT NULL,
    UNIQUE (game, draw_date));
CREATE INDEX IF NOT EXISTS ix_draws_game_number ON draws(game, draw_number);
CREATE TABLE IF NOT EXISTS toto_results (
    draw_id INTEGER PRIMARY KEY REFERENCES draws(draw_id) ON DELETE CASCADE,
    draw_number INTEGER, draw_date TEXT NOT NULL,
    n1 INTEGER, n2 INTEGER, n3 INTEGER, n4 INTEGER, n5 INTEGER, n6 INTEGER, additional INTEGER,
    {", ".join(f"{c} REAL" for c in TOTO_PRIZE_COLS)});
CREATE TABLE IF NOT EXISTS fourd_results (
    draw_id INTEGER PRIMARY KEY REFERENCES draws(draw_id) ON DELETE CASCADE,
    draw_number INTEGER, draw_date TEXT NOT NULL,
    {", ".join(f"{c} TEXT" for c in FOURD_COLS)});
CREATE TABLE IF NOT EXISTS draw_numbers (
    draw_id INTEGER NOT NULL REFERENCES draws(draw_id) ON DELETE CASCADE,
    game TEXT NOT NULL, draw_number INTEGER, draw_date TEXT NOT NULL,
    prize TEXT NOT NULL,                 -- toto: main / additional; 4d: 1st 2nd 3rd starter consolation
    position INTEGER NOT NULL,
    number TEXT NOT NULL,                -- zero-padded: '07' (TOTO), '0725' (4D)
    number_int INTEGER NOT NULL,
    digit_sum INTEGER NOT NULL, root INTEGER NOT NULL, axis TEXT NOT NULL, mirror TEXT NOT NULL,
    front TEXT, back TEXT,               -- 4D only: first and last two digits
    PRIMARY KEY (draw_id, prize, position));
CREATE INDEX IF NOT EXISTS ix_dn_game_number ON draw_numbers(game, number);
CREATE INDEX IF NOT EXISTS ix_dn_game_axis ON draw_numbers(game, axis);
CREATE INDEX IF NOT EXISTS ix_dn_game_mirror ON draw_numbers(game, mirror);
CREATE INDEX IF NOT EXISTS ix_dn_game_date ON draw_numbers(game, draw_date);
CREATE TABLE IF NOT EXISTS number_metrics (
    number TEXT PRIMARY KEY, value INTEGER NOT NULL, digit_sum INTEGER NOT NULL, root INTEGER NOT NULL,
    axis TEXT NOT NULL, mirror TEXT NOT NULL, in_chart INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS etl_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, game TEXT NOT NULL, started_at TEXT, finished_at TEXT,
    ok INTEGER, added INTEGER, n_draws INTEGER, last_draw_date TEXT, message TEXT);
CREATE VIEW IF NOT EXISTS v_draw_number_metrics AS
SELECT dn.*,
       f.root AS front_root, f.axis AS front_axis, f.mirror AS front_mirror,
       b.root AS back_root,  b.axis AS back_axis,  b.mirror AS back_mirror
FROM draw_numbers dn
LEFT JOIN number_metrics f ON f.number = dn.front
LEFT JOIN number_metrics b ON b.number = dn.back;
"""


def _p(path) -> Path:
    return Path(path) if path else DB_PATH


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(_p(path))
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init_db(path: Path | str | None = None) -> Path:
    path = _p(path)
    with closing(connect(path)) as con, con:
        con.executescript(SCHEMA)
        con.executemany("INSERT OR REPLACE INTO games VALUES (?,?,?,?)",
                        [(g, *v) for g, v in GAMES.items()])
        con.executemany(
            "INSERT OR REPLACE INTO number_metrics VALUES (:number,:value,:digit_sum,:root,:axis,:mirror,:in_chart)",
            reference_chart())
    return path


# --------------------------------------------------------------------------------------------- loading
def _iso(dates: pd.Series) -> pd.Series:
    from .data.ingest import _parse_dates
    return _parse_dates(dates)


def _num(v):
    v = str(v).strip().replace(",", "")
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return None


def _replace_game(con, game: str, frame: pd.DataFrame, table: str, cols: list[str], numbers) -> int:
    """Delete the game's rows and insert ``frame`` (one row per draw). ``numbers(row)`` yields
    (prize, position, number_text) for draw_numbers."""
    con.execute("DELETE FROM draws WHERE game = ?", (game,))  # cascades to results + draw_numbers
    width = GAMES[game][1]
    n = 0
    for r in frame.to_dict("records"):
        d = r["_date"]
        dn = int(r["draw_number"]) if str(r.get("draw_number", "")).strip().isdigit() else None
        cur = con.execute("INSERT INTO draws (game, draw_number, draw_date, weekday) VALUES (?,?,?,?)",
                          (game, dn, d.strftime("%Y-%m-%d"), d.strftime("%a")))
        draw_id = cur.lastrowid
        values = [r[c] for c in cols]
        con.execute(f"INSERT INTO {table} (draw_id, draw_number, draw_date, {', '.join(cols)}) "
                    f"VALUES ({', '.join('?' * (len(cols) + 3))})", (draw_id, dn, d.strftime("%Y-%m-%d"), *values))
        rows = []
        for prize, pos, num in numbers(r):
            m = metrics(num, width)
            rows.append((draw_id, game, dn, d.strftime("%Y-%m-%d"), prize, pos, m["number"], int(m["number"]),
                         m["digit_sum"], m["root"], m["axis"], m["mirror"], m.get("front"), m.get("back")))
        con.executemany("INSERT INTO draw_numbers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        n += 1
    return n


def sync_toto(csv_path: Path | str | None = None, db_path: Path | str | None = None) -> int:
    """Rebuild the TOTO rows from toto_results.csv (all columns, including prize data)."""
    from .store import STORE_PATH
    csv_path = Path(csv_path or STORE_PATH)
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    raw = raw.assign(_date=_iso(raw["draw_date"])).dropna(subset=["_date"])
    raw = raw.drop_duplicates("_date", keep="last").sort_values("_date")
    for c in ["draw_number", *TOTO_PRIZE_COLS]:
        if c not in raw:
            raw[c] = ""
    main = [f"n{i}" for i in range(1, 7)]
    frame = raw[["draw_number", "_date", *main, "additional", *TOTO_PRIZE_COLS]].copy()
    for c in main + ["additional"] + TOTO_PRIZE_COLS:
        frame[c] = [_num(v) for v in frame[c]]
    frame = frame.astype(object).where(frame.notna(), None)

    def numbers(r):
        for i, c in enumerate(sorted(int(r[c]) for c in main), 1):
            yield "main", i, c
        if r["additional"] is not None:
            yield "additional", 1, int(r["additional"])

    init_db(db_path)
    with closing(connect(db_path)) as con, con:
        return _replace_game(con, "toto", frame, "toto_results", main + ["additional"] + TOTO_PRIZE_COLS, numbers)


def read_fourd_csv(csv_path: Path | str | None = None) -> pd.DataFrame:
    path = Path(csv_path or FOURD_PATH)
    if not path.exists():
        return pd.DataFrame(columns=["draw_number", "draw_date", *FOURD_COLS])
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in FOURD_COLS:
        s = raw[c].astype(str).str.strip() if c in raw else pd.Series("", index=raw.index)
        raw[c] = s.where(s == "", s.str.zfill(4))  # 18 -> 0018 if a spreadsheet dropped the zeros
    return raw


def sync_fourd(csv_path: Path | str | None = None, db_path: Path | str | None = None) -> int:
    """Rebuild the 4D rows from fourd_results.csv."""
    raw = read_fourd_csv(csv_path)
    raw = raw.assign(_date=_iso(raw["draw_date"])).dropna(subset=["_date"])
    raw = raw.drop_duplicates("_date", keep="last").sort_values("_date")
    frame = raw[["draw_number", "_date", *FOURD_COLS]].copy()
    frame = frame.astype(object).where(frame != "", None)

    def numbers(r):
        for prize, c in (("1st", "first"), ("2nd", "second"), ("3rd", "third")):
            if r[c]:
                yield prize, 1, r[c]
        for prize, cols in (("starter", FOURD_STARTER), ("consolation", FOURD_CONSOLATION)):
            for i, c in enumerate(cols, 1):
                if r[c]:
                    yield prize, i, r[c]

    init_db(db_path)
    with closing(connect(db_path)) as con, con:
        return _replace_game(con, "4d", frame, "fourd_results", FOURD_COLS, numbers)


def sync_all(toto_csv=None, fourd_csv=None, db_path=None) -> dict:
    """Load both CSV files into the database. Returns draws stored per game."""
    out = {"toto": sync_toto(toto_csv, db_path)}
    out["4d"] = sync_fourd(fourd_csv, db_path)
    return out


def log_etl_run(game: str, started_at: str, ok: bool, added: int, n_draws: int, last_draw_date: str | None,
                message: str = "", db_path=None) -> None:
    init_db(db_path)
    with closing(connect(db_path)) as con, con:
        con.execute("INSERT INTO etl_runs (game, started_at, finished_at, ok, added, n_draws, last_draw_date, message) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (game, started_at, now_iso(), int(ok), int(added), int(n_draws), last_draw_date, message))


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_db(db_path=None, toto_csv=None, fourd_csv=None) -> Path:
    """Create and fill the database once if it does not exist yet (or has no draws)."""
    path = _p(db_path)
    if path.exists():
        with closing(connect(path)) as con:
            try:
                if con.execute("SELECT COUNT(*) FROM draws").fetchone()[0]:
                    return path
            except sqlite3.OperationalError:
                pass
    sync_all(toto_csv, fourd_csv, path)
    return path


# --------------------------------------------------------------------------------------------- queries
def query(sql: str, params=(), db_path=None) -> pd.DataFrame:
    with closing(connect(db_path)) as con:
        return pd.read_sql_query(sql, con, params=params)


def summary(db_path=None) -> pd.DataFrame:
    """Draws, first and last date per game."""
    return query("SELECT g.game, g.name, COUNT(d.draw_id) AS draws, MIN(d.draw_date) AS first_date, "
                 "MAX(d.draw_date) AS last_date, MAX(d.draw_number) AS last_draw_number "
                 "FROM games g LEFT JOIN draws d ON d.game = g.game GROUP BY g.game ORDER BY g.game DESC",
                 db_path=db_path)


def table_counts(db_path=None) -> pd.DataFrame:
    tables = ["games", "draws", "toto_results", "fourd_results", "draw_numbers", "number_metrics", "etl_runs"]
    with closing(connect(db_path)) as con:
        return pd.DataFrame([(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in tables],
                            columns=["table", "rows"])


def fourd_draws(db_path=None) -> pd.DataFrame:
    return query("SELECT * FROM fourd_results ORDER BY draw_date DESC", db_path=db_path)


def draw_numbers(game: str, prizes: list[str] | None = None, date_from: str | None = None,
                 date_to: str | None = None, db_path=None) -> pd.DataFrame:
    sql = "SELECT * FROM v_draw_number_metrics WHERE game = ?"
    params: list = [game]
    if prizes:
        sql += f" AND prize IN ({', '.join('?' * len(prizes))})"
        params += list(prizes)
    if date_from:
        sql += " AND draw_date >= ?"
        params.append(str(date_from))
    if date_to:
        sql += " AND draw_date <= ?"
        params.append(str(date_to))
    return query(sql + " ORDER BY draw_date DESC, prize, position", params, db_path=db_path)


def lookup(game: str, number: str, db_path=None) -> dict:
    """Where a number, its axis and its mirror were drawn before (same game)."""
    m = metrics(number, GAMES[game][1])
    out = {"metrics": m}
    for key in ("number", "axis", "mirror"):
        out[key] = query("SELECT draw_date, draw_number, prize, position, number FROM draw_numbers "
                         "WHERE game = ? AND number = ? ORDER BY draw_date DESC", (game, m[key]), db_path=db_path)
    if game == "4d":
        perm = "".join(sorted(m["number"]))
        df = query("SELECT draw_date, draw_number, prize, number FROM draw_numbers WHERE game = '4d'", db_path=db_path)
        out["permutation"] = df[df["number"].map(lambda s: "".join(sorted(s)) == perm)].sort_values(
            "draw_date", ascending=False)
    return out


def chart(db_path=None, only_document: bool = True) -> pd.DataFrame:
    sql = "SELECT number, root, axis, mirror FROM number_metrics"
    return query(sql + (" WHERE in_chart = 1" if only_document else "") + " ORDER BY value", db_path=db_path)


def etl_history(limit: int = 20, db_path=None) -> pd.DataFrame:
    return query("SELECT game, started_at, finished_at, ok, added, n_draws, last_draw_date, message "
                 "FROM etl_runs ORDER BY run_id DESC LIMIT ?", (limit,), db_path=db_path)
