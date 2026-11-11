# Real-Time Fraud Detection Platform 

- Pandas feature engineering (transaction velocity, amount vs. historical card average, device/address changes)
- XGBoost training with class weighting and chronological validation
- MLflow experiment tracking (optional)
- FastAPI fraud-scoring API
- Kafka transaction replay and a FastAPI scoring consumer
- Feast card history in a local SQLite online store
- Calibrated PyTorch Geometric graph scores blended with XGBoost
- Evidently drift monitoring with automatic XGBoost candidate training
- React analyst dashboard with durable alert cases and feedback
- Docker Compose for local development and Kubernetes manifests



## 1. Dataset

Download and place these files in `data/raw/`:
- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`

The training transaction file contains `isFraud`. 

## 2. Python setup

Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Train a baseline model

```bash
python -m src.data.prepare
python -m src.models.train
```

The training script:
- reads prepared training data,
- uses a chronological split based on `TransactionDT`,
- trains XGBoost with `scale_pos_weight`,
- reports PR-AUC, ROC-AUC, precision, recall and F1,
- saves `artifacts/xgb_model.joblib` and `artifacts/feature_columns.json`,
- logs metrics to MLflow when the tracking server/client is available.

To view MLflow locally:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

## 4. Run the API

```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger: http://localhost:8000/docs

Example request:

```bash
curl -X POST http://localhost:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"transaction_id":"demo-001","TransactionAmt":125.50,"TransactionDT":86400,"card1":12345,"addr1":315,"DeviceInfo":"ExamplePhone"}'
```

The API needs a trained model. Until a model exists, `/health` works and `/score` returns a clear model-not-found error.

## 5. Replay transactions through Kafka

Start the broker in Docker and run FastAPI with its Kafka worker enabled in a separate terminal:

```bash
docker compose up -d kafka
KAFKA_ENABLED=true python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

```bash
python -m src.streaming.producer --input data/processed/train.parquet --limit 1000
```

The replay sorts by `TransactionDT`, starts at the last 20% by default, and strips ground-truth labels. Use `--start-time` to choose another point in the dataset. The FastAPI worker consumes `transactions`, applies the same feature lookup and scoring as `POST /score`, and publishes to `fraud-scores`. Invalid events go to `fraud-errors`. The worker commits an offset after the broker acknowledges the output. Scores and alert cases are keyed by transaction ID, so replay retries reuse them.

## 6. Materialize Feast card history

```bash
python -m src.features.card_history
```

The command materializes a chronological card-proxy snapshot in SQLite. When the transaction is at or after the snapshot cutoff, the API looks up the historical count and mean to populate the matching XGBoost inputs and return card context. [Local Feast repository](feast/README.md)

## 7. Train the account–card–device GNN

```bash
python -m pip install -r requirements-graph.txt
python -m src.graph.rings
```

The training job builds a typed graph from the earliest 60% of transactions. Two GraphSAGE layers learn from links among accounts, cards, and devices. Very broad proxy hubs are excluded from the links. The job trains on the next 20%, calibrates on the next 10%, and evaluates on the final 10%. It saves an auditable PyTorch checkpoint and a CPU serving export in `artifacts/`.

Provide tokenized `account_id`, `card_id`, and `device_id` columns when available. For IEEE-CIS the fallbacks are `card1 + addr1`, `card1`, and `DeviceInfo`. The dataset has no confirmed ring labels. The graph model uses fraud across distinct accounts in a candidate component as weak supervision. A source with confirmed binary labels can supply `ring_label`. Ring IDs identify investigation leads from the historical graph snapshot.

The API uses a 70% XGBoost and 30% calibrated GNN blend for transactions connected to the historical graph. Requests with no usable graph features use XGBoost. Restart FastAPI after rebuilding the graph export or Feast snapshot.

## 8. Drift and retraining

```bash
python -m src.monitoring.retrain
```

The monitor compares up to 1,000 recent scored transactions with the training reference. It writes Evidently HTML and JSON reports under `FRAUD_STATE_DIR` (default `artifacts/state`). The Kubernetes CronJob runs every 15 minutes. It checks after at least 100 events and triggers candidate XGBoost training when at least half the compared features drift and at least 20 closed cases supply usable labels. Training and evaluation use disjoint chronological feedback windows with both classes. The candidate is compared with the active XGBoost model by average precision and saved for review. The live model remains in service while the candidate is reviewed.

For a one-off report using explicit CSV or Parquet windows, run `python -m src.monitoring.drift_report --reference REF --current CURRENT`.

## 9. Analyst dashboard and Kubernetes

For local development run `npm ci` and `npm run dev` in `dashboard/`, then open `http://localhost:5173`. The dashboard shows the blended decision, candidate ring links, the latest drift report, and a persistent queue of REVIEW cases. Analysts can investigate or close a case and record a fraud or legitimate label. Closed cases supply retraining feedback.

Build and deploy to a local cluster after preparing the data and training/materializing the artifacts:

```bash
# Use Minikube, or create a k3d cluster named fraud-platform and select its context.
minikube start --driver=docker
./scripts/deploy-local.sh
kubectl -n fraud-platform port-forward svc/fraud-dashboard 8080:8080
```

Open `http://localhost:8080`. Kubernetes runs Kafka, the FastAPI scorer/consumer, the React dashboard, a replay Job, and the drift/retraining CronJob. The deploy script copies the prepared training Parquet into a persistent volume. A separate volume retains scores, cases, reports, and candidates across restarts. The local stack is reached through a loopback port-forward.

Run the monitor immediately after replay with `kubectl -n fraud-platform create job --from=cronjob/fraud-drift-monitor drift-once`.

## 10. Project layout

```text
fraud-detection-platform/
├── data/{raw,processed}/
├── artifacts/
├── dashboard/
├── feast/
├── k8s/
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── graph/
│   ├── models/
│   ├── monitoring/
│   └── streaming/
├── tests/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Important production safeguards

- Use time-based splits and point-in-time feature joins.
- Do not use `TransactionID` or post-outcome fields as predictive features.
- Tune the alert threshold against investigation capacity and false-positive cost.
- Keep a human review path for high-impact decisions.
- Protect PII, credentials and transaction data; add authentication, authorization, encryption, audit logs and retention controls before deployment.
- Do not automatically retrain and deploy a model solely because a drift metric crossed a threshold. Require evaluation gates.
