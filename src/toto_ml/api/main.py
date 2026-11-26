"""Optional JSON API over the trained artifact and the results store (the Streamlit app does not need it).

Run:  uv run toto-ml api      ->  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..artifact import DEFAULT_ARTIFACT_DIR, load_artifact, predict_next
from ..data import DataValidationError, ingest_bytes
from ..logging_setup import setup_logging
from .. import store

setup_logging()
log = logging.getLogger("toto_ml.api")
MAX_UPLOAD = 20 * 1024 * 1024

app = FastAPI(title="DrawSense API", version=__version__)
_cache: dict = {"mtime": None, "bundle": None}
_lock = threading.Lock()


def artifact_dir() -> Path:
    return Path(os.environ.get("TOTO_ML_ARTIFACT", DEFAULT_ARTIFACT_DIR))


def bundle() -> dict:
    """Load the artifact once; reload automatically when the notebook re-exports it."""
    f = artifact_dir() / "model.joblib"
    if not f.exists():
        raise HTTPException(503, f"No trained model found at {f}. Run notebook.ipynb to train and export it.")
    with _lock:
        m = f.stat().st_mtime
        if _cache["mtime"] != m:
            _cache["bundle"] = load_artifact(artifact_dir())
            _cache["mtime"] = m
        return _cache["bundle"]


@app.exception_handler(DataValidationError)
async def _dv(_, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__, "artifact": (artifact_dir() / "model.joblib").exists()}


@app.get("/api/model")
def model_info():
    return bundle()["metadata"]


@app.post("/api/predict")
async def predict(n_split: int = Form(2), n_full: int = Form(2), file: UploadFile | None = File(None)):
    """Next ranked sets: n_split from the train/test split model, then n_full from the 100% data model.
    History = the results store, or an uploaded file (draw_date, n1..n6, additional)."""
    if not (0 <= n_split <= 10 and 0 <= n_full <= 10 and 1 <= n_split + n_full <= 10):
        raise HTTPException(400, "n_split and n_full must be 0-10 with 1-10 sets in total")
    b = bundle()
    report = None
    if file is not None and file.filename:
        content = await file.read()
        if len(content) > MAX_UPLOAD:
            raise HTTPException(413, "File too large (max 20 MB)")
        res = ingest_bytes(content, file.filename)
        draws, report = res.draws, {k: v for k, v in res.report.items() if k != "excluded_rows"}
    else:
        draws = store.load_store()
    need = int(b["config"]["split"]["min_history"]) + 1
    if len(draws) < need:
        raise HTTPException(400, f"Need at least {need} valid draws (got {len(draws)}).")
    out = predict_next(b, draws, n_split=n_split, n_full=n_full)
    out["upload_report"] = report
    return out


class NewDraw(BaseModel):
    draw_date: date
    numbers: list[int] = Field(min_length=6, max_length=6)
    additional: int


@app.get("/api/draws")
def list_draws(limit: int = 20):
    d = store.load_store().tail(limit).iloc[::-1].copy()
    d["draw_date"] = d["draw_date"].dt.strftime("%Y-%m-%d")
    return d.drop(columns=["draw_number"]).to_dict(orient="records")


@app.post("/api/draws")
def add_draw(req: NewDraw):
    try:
        store.add_draw(req.draw_date, req.numbers, req.additional)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"added": req.model_dump(mode="json")}
