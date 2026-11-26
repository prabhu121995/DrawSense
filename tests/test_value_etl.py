"""Prize value sets, the stored next move, and the ETL (with a stand-in scraper, no internet)."""
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from toto_ml.value import fit_popularity, shape_ok, shape_profile, value_index, value_sets

REAL = pytest.importorskip("toto_ml.store").STORE_PATH


def _prize_frame(n=200, seed=0):
    """Synthetic results where draws with more numbers <= 31 have more winners (as in real data)."""
    rng = np.random.default_rng(seed)
    X = np.sort(np.argsort(rng.random((n, 49)), axis=1)[:, :6] + 1, axis=1)
    low = (X <= 31).sum(1)
    crowd = np.exp(0.15 * low + rng.normal(0, 0.05, n))
    df = pd.DataFrame(X, columns=[f"n{i}" for i in range(1, 7)])
    df["g4_share_amount"], df["g4_winning_shares"] = 400.0, 300.0
    df["g3_share_amount"] = 2000 / crowd
    w = crowd * 1000
    df["g5_winning_shares"], df["g6_winning_shares"], df["g7_winning_shares"] = w * .05, w * .1, w
    return df


def test_popularity_learns_birthday_numbers():
    pop = fit_popularity(raw=_prize_frame())
    w = np.array(pop["weights"])
    assert pop["source"] == "prize data" and pop["cv_rho"] > 0.8
    assert w[:31].mean() > w[31:].mean()                      # 1-31 are more popular
    assert value_index([40, 42, 44, 45, 47, 49], pop) > 1 > value_index([1, 3, 7, 12, 19, 25], pop)
    assert fit_popularity(raw=pd.DataFrame({"x": [1]}))["source"] == "birthday prior"


def test_value_sets_shape_overlap_and_order(synth_draws):
    pop = fit_popularity(raw=_prize_frame())
    prof = shape_profile(synth_draws)
    p = np.random.default_rng(1).uniform(0.11, 0.13, 49)
    first = value_sets(p, p, pop, prof, n_sets=2, pool=18)
    more = value_sets(p, p, pop, prof, n_sets=2, pool=18, earlier=[s["numbers"] for s in first])
    sets = first + more
    assert len(sets) == 4
    for i, s in enumerate(sets):
        assert s["typical_shape"] and shape_ok(np.array(s["numbers"]), prof)[0]
        top18 = set(int(x) + 1 for x in np.argsort(-p)[:18])
        assert set(s["numbers"]) <= top18                      # built from the model's top numbers
        for t in sets[:i]:
            assert len(set(s["numbers"]) & set(t["numbers"])) <= 3
    assert first[0]["value_index"] >= first[1]["value_index"] - 0.01


def test_next_move_saved_and_detected_as_current(tmp_path, synth_draws):
    from toto_ml.nextmove import is_current, load_next_move, save_next_move
    last = synth_draws["draw_date"].iloc[-1].strftime("%Y-%m-%d")
    pred = {"model": "m", "next_draw_date": "2030-01-03", "strategy": "value",
            "based_on": {"n_draws": len(synth_draws), "last_draw_date": last, "last_numbers": [1, 2, 3, 4, 5, 6],
                         "last_additional": 7},
            "sets": [{"rank": 1, "source": "split", "numbers": [1, 12, 23, 34, 40, 45], "value_pct": 12.3,
                      "typical_shape": True, "odd": 3, "even": 3, "sum": 155, "mean_probability": 0.12}],
            "numbers": [{"rank": 1, "number": 45, "value_pct": 5.0}]}
    path = save_next_move(pred, tmp_path / "next_move.csv")
    df, meta = load_next_move(path)
    assert df.loc[0, "model"] == "Validated model" and df.loc[0, "prize_value_pct"] == 12.3
    assert is_current(meta, synth_draws) and not is_current(meta, synth_draws.iloc[:-1])


def test_etl_adds_new_rows_and_replaces_hand_typed_row(tmp_path, monkeypatch):
    from toto_ml import etl
    csv = tmp_path / "toto_results.csv"
    head = "draw_number,draw_date,n1,n2,n3,n4,n5,n6,additional,group1_prize\n"
    rows = [f'{4000 + i},"Mon, {i + 1:02d} Jan 2024",{i % 40 + 1},{i % 40 + 2},{i % 40 + 3},{i % 40 + 4},'
            f'{i % 40 + 5},{i % 40 + 6},49,1000000\n' for i in range(25)]
    hand = ',"Mon, 26 Jan 2024",1,2,3,4,5,7,49,\n'                 # typed in the app: no draw number
    csv.write_text(head + "".join(rows) + hand, encoding="utf-8")
    fake = tmp_path / "fake_scraper.py"
    fake.write_text(textwrap.dedent('''
        import sys
        out = sys.argv[sys.argv.index("-o") + 1]
        with open(out, "a", encoding="utf-8") as f:
            f.write('4025,"Mon, 26 Jan 2024",1,2,3,4,5,8,49,2000000\\n')   # official version of the typed draw
            f.write('4026,"Thu, 29 Jan 2024",10,20,30,40,41,42,1,3000000\\n')
        print("[1/2] #4025 saved")
    '''), encoding="utf-8")
    monkeypatch.setattr(etl, "SCRAPER", fake)
    lines = []
    res = etl.run_etl(csv, on_line=lines.append, update_model=False)
    assert res["ok"] and res["added"] == 1 and res["merged_manual_rows"] == 1
    raw = pd.read_csv(csv)
    jan26 = raw[raw["draw_date"] == "Mon, 26 Jan 2024"]
    assert len(jan26) == 1 and int(jan26["draw_number"].iloc[0]) == 4025 and int(jan26["n6"].iloc[0]) == 8
    assert any("#4025" in s for s in lines)
    assert (tmp_path / "toto_results.backup.csv").exists()
