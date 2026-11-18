import pandas as pd
import pytest

from toto_ml.data import DataValidationError, generate_synthetic_draws, ingest_bytes, validate_and_clean
from toto_ml.data.schema import OUTPUT_COLS

from .conftest import REAL_CSV


def _base(n=5):
    return generate_synthetic_draws(n, seed=1)


def test_only_eight_columns_needed_and_draw_number_generated():
    df, rep, _ = validate_and_clean(_base(20))
    assert list(df.columns) == OUTPUT_COLS
    assert df["draw_number"].tolist() == list(range(1, 21))
    assert rep["n_excluded"] == 0
    assert (df[["n1", "n2", "n3", "n4", "n5", "n6"]].diff(axis=1).iloc[:, 1:] > 0).all().all()


def test_extra_columns_ignored():
    d = _base(10).assign(group1_prize=123.0, foo="x")
    df, rep, _ = validate_and_clean(d)
    assert list(df.columns) == OUTPUT_COLS
    assert any("Ignored columns" in w for w in rep["warnings"])


def test_missing_required_column_raises():
    with pytest.raises(DataValidationError):
        validate_and_clean(_base().drop(columns=["additional"]))


@pytest.mark.parametrize("mutate,reason", [
    (lambda d: d.assign(n1=d["n1"].where(d.index != 0, 50)), "outside"),
    (lambda d: d.assign(n2=d["n2"].where(d.index != 0, d.loc[0, "n1"])), "not unique"),
    (lambda d: d.assign(additional=d["additional"].where(d.index != 0, d.loc[0, "n1"])), "repeats"),
    (lambda d: d.assign(additional=d["additional"].where(d.index != 0, 0)), "additional number outside"),
    (lambda d: d.assign(draw_date=d["draw_date"].where(d.index != 0, "not a date")), "draw_date"),
    (lambda d: d.assign(n4=d["n4"].astype(object).where(d.index != 0, None)), "missing main"),
])
def test_invalid_rows_excluded_with_reason(mutate, reason):
    df, rep, exc = validate_and_clean(mutate(_base()))
    assert len(df) == 4
    assert any(reason in e["reason"] for e in exc)


def test_duplicates_detected():
    d = _base()
    df, rep, _ = validate_and_clean(pd.concat([d, d.iloc[[1]]], ignore_index=True))
    assert len(df) == 5 and rep["n_exact_duplicates"] == 1
    conflict = d.copy()
    conflict.loc[2, "draw_date"] = conflict.loc[1, "draw_date"]
    df, rep, _ = validate_and_clean(conflict)
    assert len(df) == 3 and rep["n_conflicting_duplicates"] == 2


def test_sorted_chronologically():
    df, _, _ = validate_and_clean(_base(8).iloc[::-1])
    assert df["draw_date"].is_monotonic_increasing


def test_iso_dates_are_never_read_day_first():
    """Regression: 2026-11-12 must stay 12 November (not 11 December)."""
    d = _base(3)
    d["draw_date"] = ["2026-11-09", "2026-11-12", "2026-11-16"]
    df, _, _ = validate_and_clean(d)
    assert df["draw_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-11-09", "2026-11-12", "2026-11-16"]
    d["draw_date"] = ["09/11/2026", "12/11/2026", "16/11/2026"]  # Singapore day/month/year
    df, _, _ = validate_and_clean(d)
    assert df["draw_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-11-09", "2026-11-12", "2026-11-16"]


def test_real_file():
    if not REAL_CSV.exists():
        pytest.skip("real CSV not present")
    res = ingest_bytes(REAL_CSV.read_bytes(), REAL_CSV.name)
    assert res.report["n_valid_draws"] > 100
    assert res.draws["draw_date"].is_monotonic_increasing


def test_xlsx(tmp_path):
    p = tmp_path / "x.xlsx"
    _base(10).to_excel(p, index=False)
    assert ingest_bytes(p.read_bytes(), "x.xlsx").report["n_valid_draws"] == 10
