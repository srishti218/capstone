from contextlib import asynccontextmanager
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
from threading import Event, Thread
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api import store
from src.api.schemas import CaseStatus, CaseUpdate, FraudScore, Transaction
from src.features.serving import scoring_features
from src.models.ensemble import blend_scores


MODEL_PATH = Path("artifacts/xgb_model.joblib")
COLUMNS_PATH = Path("artifacts/feature_columns.json")
RING_PATH = Path("artifacts/fraud_ring_gnn.npz")
GRAPH_PATH = Path("artifacts/fraud_gnn.pt")
FEAST_REPO = Path(__file__).resolve().parents[2] / "feast"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    worker = None
    stop = Event()
    if RING_PATH.is_file():
        try:
            graph_scorer(RING_PATH)
        except Exception:
            logger.exception("Could not load the ring GNN at startup")
    if os.getenv("KAFKA_ENABLED", "false").lower() == "true":
        from src.streaming.consumer import consume

        worker = Thread(target=consume, args=(score, stop), daemon=True)
        worker.start()
    try:
        yield
    finally:
        stop.set()
        if worker:
            worker.join(timeout=5)


app = FastAPI(title="Fraud Detection API", version="1.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://{host}:{port}" for host in ("localhost", "127.0.0.1") for port in (5173, 4173)],
    allow_methods=["GET", "POST", "PATCH"],
)


@lru_cache(maxsize=1)
def card_store():
    from feast import FeatureStore

    return FeatureStore(repo_path=str(FEAST_REPO))


@lru_cache(maxsize=1)
def graph_scorer(path):
    if path == RING_PATH:
        from src.graph.scoring import RingScorer

        return RingScorer(path)
    from src.graph.train_gnn import GraphScorer

    return GraphScorer(path)


def scoring_context(tx, payload):
    context = {}
    feast_data = FEAST_REPO / "data"
    if tx.card1 is not None and all((feast_data / name).is_file() for name in ("registry.db", "online_store.db")):
        try:
            from src.features.card_history import lookup

            context["card_history"] = lookup(card_store(), tx.card1, tx.TransactionDT)
        except Exception:
            logger.exception("Feast card history lookup failed")
    try:
        if RING_PATH.is_file():
            context.update(graph_scorer(RING_PATH).score(payload))
        elif GRAPH_PATH.is_file():
            context["gnn_score"] = graph_scorer(GRAPH_PATH).score(payload)
    except Exception:
        logger.exception("GNN scoring failed")
    return context


def load_model():
    if not MODEL_PATH.exists() or not COLUMNS_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    model.set_params(n_jobs=2)
    return model, json.loads(COLUMNS_PATH.read_text())


model, feature_columns = load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/ready")
def ready():
    if model is None:
        raise HTTPException(status_code=503, detail="XGBoost model is unavailable.")
    return {"status": "ready"}


def drift_path():
    monitored = store.state_dir() / "drift_report.html"
    return monitored if monitored.is_file() and monitored.with_suffix(".json").is_file() else Path("artifacts/drift_report.html")


@app.get("/monitoring/drift")
def drift_summary():
    path = drift_path()
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise HTTPException(status_code=404, detail="Drift report unavailable.")
    return json.loads(path.with_suffix(".json").read_text())


@app.get("/monitoring/drift/report", response_class=FileResponse)
def drift_report():
    path = drift_path()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Drift report unavailable.")
    return FileResponse(path, media_type="text/html", headers={"Content-Security-Policy": "sandbox allow-scripts"})


@app.get("/monitoring/retraining")
def retraining_status():
    path = store.state_dir() / "retraining.json"
    return json.loads(path.read_text()) if path.is_file() else {"status": "waiting_for_monitor"}


@app.get("/cases")
def list_cases(status: Optional[CaseStatus] = None, limit: int = Query(default=100, ge=1, le=250)):
    return store.cases(status, limit)


@app.patch("/cases/{transaction_id:path}")
def patch_case(transaction_id: str, update: CaseUpdate):
    try:
        result = store.update_case(transaction_id, update.status, update.label, "label" in update.model_fields_set)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Alert case not found.")
    return result


@app.post("/score", response_model=FraudScore, response_model_exclude_none=True)
def score(tx: Transaction):
    event = tx.model_dump(exclude={"transaction_id"})
    try:
        previous = store.existing_score(tx.transaction_id, event)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if previous is not None:
        return FraudScore(**previous)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Train it with `python -m src.models.train`.")

    payload = {**event["extra_features"], **{key: value for key, value in event.items() if key != "extra_features"}}
    context = scoring_context(tx, payload)
    features = scoring_features(payload, feature_columns, context.get("card_history"))
    xgb = float(model.predict_proba(features)[0, 1])
    gnn = context.get("gnn_score") if RING_PATH.is_file() else None
    probability = blend_scores(xgb, gnn) if gnn is not None else xgb
    score = FraudScore(
        transaction_id=tx.transaction_id, fraud_probability=probability,
        risk_score=round(probability * 100), decision="REVIEW" if probability >= 0.20 else "ALLOW",
        model="xgboost+ring-gnn" if gnn is not None else "xgboost-baseline",
        xgb_probability=xgb, **context,
    )
    try:
        saved = store.save_score(tx.transaction_id, event, score.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return FraudScore(**saved)
