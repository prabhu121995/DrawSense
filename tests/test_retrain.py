"""Background retrain status handling (the actual notebook run is exercised manually / end to end)."""
import json
import time

from toto_ml import retrain


def test_idle_when_no_status(tmp_path):
    assert retrain.read_status(tmp_path / "s.json") == {"state": "idle"}


def test_running_with_fresh_heartbeat(tmp_path):
    p = tmp_path / "s.json"
    retrain._write(p, state="running", started_at=time.time(), step="x", step_t=time.time())
    assert retrain.is_running(p)
    assert retrain.start(p) is False  # a second click never starts a duplicate run


def test_stale_running_is_reported_as_failed(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"state": "running", "heartbeat": time.time() - 3600}))
    st = retrain.read_status(p)
    assert st["state"] == "failed" and "stopped unexpectedly" in st["error"]
    assert not retrain.is_running(p)


def test_training_progress_is_merged(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    retrain._write(p, state="running", started_at=time.time(), progress=0.05, step="Training", step_t=time.time() - 1)
    monkeypatch.setenv("TOTO_RETRAIN_STATUS", str(p))
    retrain.report_progress(0.5, "lightgbm: walk-forward origin 3/16")
    st = retrain.read_status(p)
    lo, hi = retrain.TRAIN_RANGE
    assert st["progress"] == round(lo + (hi - lo) * 0.5, 4)
    assert st["detail"].startswith("lightgbm")


def test_report_progress_noop_outside_retrain(tmp_path, monkeypatch):
    monkeypatch.delenv("TOTO_RETRAIN_STATUS", raising=False)
    retrain.report_progress(0.5, "x")  # must not raise or write anything
    assert not list(tmp_path.iterdir())
