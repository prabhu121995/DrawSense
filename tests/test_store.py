"""Adding future draw results to the raw results file (toto_results.csv format)."""
import io
from datetime import date

import pandas as pd
import pytest

from toto_ml import store
from toto_ml.data import generate_synthetic_draws


@pytest.fixture()
def raw_path(tmp_path):
    """A results file shaped like the scraper output: draw_number, SG dates, prize columns (some blank)."""
    d = generate_synthetic_draws(60, seed=5)
    d.insert(0, "draw_number", range(3900, 3960))
    d["group1_prize"] = 1234567
    d["g1_winning_shares"] = ""
    p = tmp_path / "toto_results.csv"
    d.to_csv(p, index=False)
    return p


def _next_date(path):
    return (store.load_store(path)["draw_date"].max() + pd.Timedelta(days=3)).date()


def test_add_appends_row_and_keeps_other_columns(raw_path):
    before_text = raw_path.read_text().splitlines()
    d = _next_date(raw_path)
    out = store.add_draw(d, [49, 3, 17, 25, 8, 40], 11, raw_path)
    assert len(out) == 61
    lines = raw_path.read_text().splitlines()
    assert lines[:-1] == before_text                       # existing rows untouched, byte for byte
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    new = raw.iloc[-1]
    assert new["draw_date"] == pd.Timestamp(d).strftime("%a, %d %b %Y")  # same date style as the file
    assert [int(new[f"n{i}"]) for i in range(1, 7)] == [3, 8, 17, 25, 40, 49]
    assert new["additional"] == "11"
    assert new["draw_number"] == "" and new["group1_prize"] == ""      # unknown fields left blank
    assert list(raw.columns) == list(pd.read_csv(store.backup_path(raw_path)).columns)


def test_backup_written_once(raw_path):
    original = raw_path.read_text()
    store.add_draw(_next_date(raw_path), [1, 2, 3, 4, 5, 6], 7, raw_path)
    store.add_draw(_next_date(raw_path), [11, 12, 13, 14, 15, 16], 17, raw_path)
    assert store.backup_path(raw_path).read_text() == original


def test_new_draw_is_latest_in_loaded_history(raw_path):
    d = _next_date(raw_path)
    store.add_draw(d, [5, 9, 13, 22, 31, 44], 2, raw_path)
    h = store.load_store(raw_path)
    assert h["draw_date"].iloc[-1].date() == d
    assert h["draw_date"].is_monotonic_increasing


@pytest.mark.parametrize("nums,add,msg", [
    ([1, 2, 3, 4, 5], 9, "exactly 6"),
    ([1, 1, 3, 4, 5, 6], 9, "different"),
    ([1, 2, 3, 4, 5, 50], 9, "between 1 and 49"),
    ([1, 2, 3, 4, 5, 6], 6, "cannot be one of the main"),
])
def test_invalid_new_draw_rejected(raw_path, nums, add, msg):
    original = raw_path.read_text()
    with pytest.raises(ValueError, match=msg):
        store.add_draw(_next_date(raw_path), nums, add, raw_path)
    assert raw_path.read_text() == original  # nothing written


def test_duplicate_or_older_date_rejected(raw_path):
    draws = store.load_store(raw_path)
    with pytest.raises(ValueError, match="already exists"):
        store.add_draw(draws["draw_date"].iloc[-1].date(), [1, 2, 3, 4, 5, 6], 7, raw_path)
    with pytest.raises(ValueError, match="before the latest"):
        store.add_draw(date(2000, 1, 1), [1, 2, 3, 4, 5, 6], 7, raw_path)


def test_merge_upload_adds_only_new_dates(raw_path):
    full = generate_synthetic_draws(65, seed=5)  # first 60 identical to the file, 5 new
    main0 = full.loc[0, ["n1", "n2", "n3", "n4", "n5", "n6"]].tolist()
    full.loc[0, "additional"] = next(x for x in range(1, 50) if x not in main0 and x != full.loc[0, "additional"])
    buf = io.BytesIO()
    full.to_csv(buf, index=False)
    res = store.merge_upload(buf.getvalue(), "more.csv", raw_path)
    assert res["added"] == 5 and res["already_present"] == 60
    assert len(res["conflicts_kept_stored_version"]) == 1  # changed row is reported, stored version kept
    assert len(store.load_store(raw_path)) == 65


def test_delete_last(raw_path):
    d = _next_date(raw_path)
    store.add_draw(d, [5, 9, 13, 22, 31, 44], 2, raw_path)
    removed = store.delete_last(raw_path)
    assert removed["draw_date"].date() == d
    assert len(store.load_store(raw_path)) == 60
