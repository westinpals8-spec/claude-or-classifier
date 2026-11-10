"""FastAPI service for app feedback triage (Week 5 deployment prototype).

Run locally (from the project root):
    export TRIAGE_API_KEY=change-me            # required by clients in the X-API-Key header
    export ANTHROPIC_API_KEY=...               # optional: enables the Claude fallback
    uvicorn src.service.app:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health            liveness check (no auth)
    GET  /v1/model-info     model version, thresholds, uncertainty band, LLM status
    POST /v1/classify       classify up to 100 reviews per call
"""
from __future__ import annotations

import logging
import os
import secrets
import time

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .predictor import DEFAULT_BAND, LABELS, MODEL_VERSION, THRESHOLDS, TriageService

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.environ.get("TRIAGE_MODEL_DIR", os.path.join(ROOT, "models", "distilbert_best"))
QUANTIZE = os.environ.get("TRIAGE_QUANTIZE", "0") == "1"
MAX_BATCH, MAX_CHARS = 100, 2000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("triage")
service = TriageService(MODEL_DIR, band=DEFAULT_BAND, quantize=QUANTIZE)
app = FastAPI(title="App Feedback Triage API", version="1.0.0")


class Review(BaseModel):
    id: str | None = Field(default=None, max_length=64)
    text: str = Field(min_length=1, max_length=MAX_CHARS)


class ClassifyRequest(BaseModel):
    reviews: list[Review] = Field(min_length=1, max_length=MAX_BATCH)


class ReviewResult(BaseModel):
    id: str | None
    labels: list[str]
    probabilities: dict[str, float]
    route: str
    model: str
    latency_ms: float
    note: str = ""


class ClassifyResponse(BaseModel):
    results: list[ReviewResult]
    counts: dict[str, int]


def require_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("TRIAGE_API_KEY")
    if not expected or not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/model-info", dependencies=[Depends(require_key)])
def model_info():
    return {"model_version": MODEL_VERSION, "labels": LABELS, "thresholds": THRESHOLDS,
            "uncertainty_band": DEFAULT_BAND, "quantized": QUANTIZE,
            "llm_fallback": service.llm.model if service.llm.available else "disabled (human review queue)"}


@app.post("/v1/classify", response_model=ClassifyResponse, dependencies=[Depends(require_key)])
def classify(req: ClassifyRequest):
    t0 = time.perf_counter()
    preds = service.classify([r.text for r in req.reviews])
    results = [ReviewResult(id=r.id, labels=p.labels, probabilities=p.probabilities, route=p.route,
                            model=p.model, latency_ms=p.latency_ms, note=p.note) for r, p in zip(req.reviews, preds)]
    counts = {k: sum(p.route == k for p in preds) for k in ("local", "llm", "human_review")}
    # privacy: log fingerprints and routes only, never review text
    log.info("classified n=%d routes=%s ms=%.1f fps=%s", len(preds), counts,
             1000 * (time.perf_counter() - t0), [p.fingerprint for p in preds[:5]])
    return ClassifyResponse(results=results, counts=counts)
