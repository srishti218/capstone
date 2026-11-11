from pathlib import Path
import json
import joblib
import pandas as pd
import mlflow
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from src.features.build_features import build_features

DATA = Path("data/processed/train.parquet")
ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

def main():
    if not DATA.exists():
        raise FileNotFoundError("Run `python -m src.data.prepare` first.")
    raw = pd.read_parquet(DATA)
    if "isFraud" not in raw:
        raise ValueError("Training data must contain isFraud.")
    raw = raw.sort_values("TransactionDT", kind="stable")
    X = build_features(raw, training=True)
    y = raw.loc[X.index, "isFraud"].astype(int)

    # Chronological holdout; no random shuffling.
    cut = int(len(X) * 0.80)
    X_train, X_valid = X.iloc[:cut], X.iloc[cut:]
    y_train, y_valid = y.iloc[:cut], y.iloc[cut:]
    pos = max(int(y_train.sum()), 1)
    neg = max(int(len(y_train) - pos), 1)
    model = XGBClassifier(
        n_estimators=700, max_depth=6, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85,
        objective="binary:logistic", eval_metric="aucpr",
        scale_pos_weight=neg / pos, tree_method="hist",
        random_state=42, n_jobs=-1
    )
    mlflow.set_experiment("ieee-cis-fraud-detection")
    with mlflow.start_run():
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        probs = model.predict_proba(X_valid)[:, 1]
        pred = (probs >= 0.5).astype(int)
        metrics = {
            "pr_auc": average_precision_score(y_valid, probs),
            "roc_auc": roc_auc_score(y_valid, probs),
            "precision_at_0_5": precision_score(y_valid, pred, zero_division=0),
            "recall_at_0_5": recall_score(y_valid, pred, zero_division=0),
            "f1_at_0_5": f1_score(y_valid, pred, zero_division=0),
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        mlflow.log_param("scale_pos_weight", neg / pos)
        mlflow.log_param("chronological_split", "80/20")
        joblib.dump(model, ARTIFACTS / "xgb_model.joblib")
        (ARTIFACTS / "feature_columns.json").write_text(json.dumps(list(X.columns), indent=2))
        mlflow.log_artifact(str(ARTIFACTS / "xgb_model.joblib"))
        print(json.dumps(metrics, indent=2))
        print(f"Saved model to {ARTIFACTS / 'xgb_model.joblib'}")

if __name__ == "__main__":
    main()
