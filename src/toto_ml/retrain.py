"""Background retraining: runs notebook.ipynb in a separate process and reports progress.

    start()         launch a retrain in the background (no-op if one is already running)
    read_status()   current state for the UI: idle | running | done | failed
    python -m toto_ml.retrain   (what start() launches) executes the notebook cell by cell

Progress lives in ``artifacts/retrain_status.json`` so any browser tab or a page refresh can
show it, and a second click can't start a duplicate run. The notebook calls
``report_progress`` from its training loop for fine-grained progress.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import PROJECT_ROOT

NOTEBOOK = PROJECT_ROOT / "notebook.ipynb"
ARTIFACTS = PROJECT_ROOT / "artifacts"
STATUS_PATH = Path(os.environ.get("TOTO_RETRAIN_STATUS", ARTIFACTS / "retrain_status.json"))
LOG_PATH = ARTIFACTS / "retrain.log"
HEARTBEAT_S = 3
STALE_AFTER_S = 45  # no heartbeat for this long => the runner died (e.g. PC restarted)
_lock = threading.Lock()


def _now() -> float:
    return time.time()


def read_status(path: Path | None = None) -> dict:
    path = Path(path or STATUS_PATH)
    try:
        st = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"state": "idle"}
    if st.get("state") == "running":
        if _now() - st.get("heartbeat", 0) > STALE_AFTER_S:
            st["state"] = "failed"
            st["error"] = "The retrain process stopped unexpectedly (no heartbeat). Start it again."
        else:  # merge fine-grained progress written by the training cell (separate file: no write races)
            try:
                pr = json.loads(_progress_path(path).read_text(encoding="utf-8"))
                if pr.get("t", 0) >= st.get("step_t", 0):
                    st["progress"], st["detail"] = pr["progress"], pr["detail"]
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass
    return st


def _progress_path(status_path: Path) -> Path:
    return Path(status_path).with_name("retrain_progress.json")


def _write(path: Path, **fields) -> dict:
    with _lock:
        st = read_status(path) if path.exists() else {}
        st.update(fields)
        st["heartbeat"] = _now()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, default=str), encoding="utf-8")
        os.replace(tmp, path)
        return st


def is_running(path: Path | None = None) -> bool:
    return read_status(path).get("state") == "running"


def start(path: Path | None = None) -> bool:
    """Launch a background retrain. Returns False if one is already running."""
    path = Path(path or STATUS_PATH)
    if is_running(path):
        return False
    _progress_path(path).unlink(missing_ok=True)
    _write(path, state="running", started_at=_now(), finished_at=None, progress=0.0,
           step="Starting", detail="", error=None, result=None, step_t=_now())
    env = {**os.environ, "TOTO_RETRAIN_STATUS": str(path), "PYTHONUNBUFFERED": "1"}
    env.pop("MPLBACKEND", None)  # a backend inherited from the app would stop the notebook saving its charts
    flags = 0
    if os.name == "nt":  # keep running even if the Streamlit process is stopped
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as log:
        subprocess.Popen([sys.executable, "-m", "toto_ml.retrain"], cwd=PROJECT_ROOT, env=env,
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
    return True


# ---------------------------------------------------------------- called from the notebook
TRAIN_RANGE = (0.05, 0.92)  # share of the progress bar used by the (long) training cell


def report_progress(frac: float, msg: str) -> None:
    """Fine-grained progress from inside the training cell. No-op outside a background retrain."""
    p = os.environ.get("TOTO_RETRAIN_STATUS")
    if not p:
        return
    lo, hi = TRAIN_RANGE
    try:
        pp = _progress_path(Path(p))
        tmp = pp.with_suffix(".tmp")
        tmp.write_text(json.dumps({"progress": round(lo + (hi - lo) * float(frac), 4), "detail": msg, "t": _now()}),
                       encoding="utf-8")
        os.replace(tmp, pp)
    except OSError:
        pass


# ------------------------------------------------------------------------- runner process
def _run() -> int:
    import nbformat
    from nbclient import NotebookClient

    path = STATUS_PATH
    nb = nbformat.read(NOTEBOOK, as_version=4)
    cells = nb.cells
    n = len(cells)
    train_idx = next((i for i, c in enumerate(cells) if c.cell_type == "code" and "run_experiment(" in c.source), None)
    headings = {}
    current = "Preparing"
    for i, c in enumerate(cells):
        if c.cell_type == "markdown" and c.source.lstrip().startswith("#"):
            current = c.source.lstrip("# ").splitlines()[0].strip()
        headings[i] = current

    stop = threading.Event()

    def beat():
        while not stop.wait(HEARTBEAT_S):
            try:
                _write(path)
            except OSError:
                pass

    threading.Thread(target=beat, daemon=True).start()

    def on_start(cell, cell_index, **_):
        if train_idx is None:
            prog = cell_index / n
        elif cell_index < train_idx:
            prog = 0.05 * cell_index / max(train_idx, 1)
        elif cell_index == train_idx:
            prog = 0.05
        else:
            prog = 0.92 + 0.08 * (cell_index - train_idx) / max(n - train_idx, 1)
        step = headings.get(cell_index, "")
        if cell_index == train_idx:
            step = "Training & benchmarking all algorithms"
        _write(path, progress=round(prog, 4), step=step, detail=f"cell {cell_index + 1}/{n}", step_t=_now())

    client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(PROJECT_ROOT)}}, on_cell_start=on_start)
    t0 = _now()
    try:
        client.execute()
        meta = json.loads((ARTIFACTS / "best_model" / "metadata.json").read_text(encoding="utf-8"))
        _write(path, state="done", progress=1.0, step="Done", detail="", finished_at=_now(),
               duration_s=round(_now() - t0), result={"best_model": meta["best_model_label"],
                                                      "n_draws": meta["training_data"]["n_draws"],
                                                      "date_max": meta["training_data"]["date_max"]})
        return 0
    except Exception as e:  # noqa: BLE001 - report every failure to the UI
        msg = str(e)
        _write(path, state="failed", finished_at=_now(), error=msg[-3000:], step="Failed")
        print(msg, file=sys.stderr)
        return 1
    finally:
        stop.set()
        try:  # save the executed notebook (outputs), also after a failure so the error is visible
            nbformat.write(nb, NOTEBOOK)
        except Exception:  # noqa: BLE001
            pass


def fmt_elapsed(seconds: float) -> str:
    s = int(max(seconds, 0))
    return f"{s // 60}m {s % 60:02d}s"


def started_at_text(st: dict) -> str:
    ts = st.get("started_at")
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%d %b %H:%M") if ts else ""


if __name__ == "__main__":
    sys.exit(_run())
