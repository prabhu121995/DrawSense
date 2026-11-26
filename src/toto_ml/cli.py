"""Command line.

    toto-ml ui                                     start the Streamlit app (http://localhost:8501)
    toto-ml predict [--split 2 --full 2]           print the next ranked sets (split model + 100% model)
    toto-ml add 2026-11-19 1 2 3 4 5 6 --additional 7   add a new draw result
    toto-ml api [--port 8000]                      optional JSON API
    toto-ml etl                                    get the latest results from the web, update model + next move

Training is done in notebook.ipynb.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from .logging_setup import setup_logging


def main(argv=None) -> int:
    setup_logging("WARNING")
    p = argparse.ArgumentParser(prog="toto-ml")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ui")
    s = sub.add_parser("predict")
    s.add_argument("--split", type=int, default=None, help="sets from the train/test split model (default 2)")
    s.add_argument("--full", type=int, default=None, help="sets from the 100%% data model (default 2)")
    s = sub.add_parser("add")
    s.add_argument("date", help="YYYY-MM-DD")
    s.add_argument("numbers", nargs=6, type=int)
    s.add_argument("--additional", type=int, required=True)
    sub.add_parser("etl")
    s = sub.add_parser("api")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)

    from .config import PROJECT_ROOT
    if args.cmd == "ui":
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "app.py")])
    if args.cmd == "api":
        import uvicorn
        uvicorn.run("toto_ml.api.main:app", host=args.host, port=args.port)
        return 0

    if args.cmd == "etl":
        from .etl import run_etl
        res = run_etl(on_line=print)
        print(f"\n{res['added']} new draw(s); {res['n_draws']} draws up to {res['last_draw_date']}. "
              f"Model updated: {res['model_updated']}. Next move updated: {res['next_move_updated']}.")
        return 0 if res["ok"] else 1

    from . import store
    if args.cmd == "add":
        from datetime import date
        try:
            draws = store.add_draw(date.fromisoformat(args.date), args.numbers, args.additional)
        except ValueError as e:
            print("Not added:", e)
            return 1
        print(f"Added. Store now has {len(draws)} draws ({store.STORE_PATH}).")
        return 0

    from .artifact import DEFAULT_ARTIFACT_DIR, load_artifact, predict_next
    r = predict_next(load_artifact(DEFAULT_ARTIFACT_DIR), store.load_store(), n_split=args.split, n_full=args.full)
    print(f"Model: {r['model']}  |  history: {r['based_on']['n_draws']} draws up to {r['based_on']['last_draw_date']}"
          f"  |  next draw: {r['next_draw_date']}\n")
    print(f"{'Rank':<6}{'Model':<20}{'Numbers':<26}{'Prize value':>12}")
    for s in r["sets"]:
        src = {"split": "Validated model", "full": "Full-history model"}[s["source"]]
        val = f"+{s['value_pct']:.0f}%" if "value_pct" in s else ""
        print(f"{s['rank']:<6}{src:<20}{'  '.join(f'{n:>2}' for n in s['numbers']):<26}{val:>12}")
    print("\nTop 10 numbers: " + ", ".join(f"{x['rank']}. {x['number']}" for x in r["numbers"][:10]))
    print(f"\n{r['disclaimer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
