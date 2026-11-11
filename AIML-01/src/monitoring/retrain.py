"""Compare recent scored traffic and train a candidate when drift and labels warrant it."""
import argparse
import fcntl
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.api import store
from src.data.parquet_rows import select_rows
from src.features.build_features import build_features
from src.features.proxy_history import chronological_cutoffs
from src.features.serving import scoring_features
from src.monitoring.drift_report import DEFAULT_COLUMNS, generate_report


def flatten(event):
    return {**event.get("extra_features", {}), **{key: value for key, value in event.items() if key != "extra_features"}}


def feedback_frame(labels):
    records = []
    for identifier, event, label, _, card_history in labels:
        fields = flatten(event)
        source_id = identifier.removeprefix("replay-")
        if identifier.startswith("replay-") and source_id.isdecimal():
            fields["TransactionID"] = int(source_id)
        fields["isFraud"] = int(label == "fraud")
        fields["_served_card_history"] = card_history
        records.append(fields)
    return pd.DataFrame(records)


def evaluate_and_train(input_path, feedback, columns, active_model, candidate_dir):
    feedback = feedback.sort_values("TransactionDT", kind="stable")
    cutoff = feedback.TransactionDT.iloc[len(feedback) * 4 // 5]
    fit = feedback.loc[feedback.TransactionDT < cutoff]
    validate = feedback.loc[feedback.TransactionDT >= cutoff]
    if fit.isFraud.nunique() != 2 or validate.isFraud.nunique() != 2:
        return None

    # Bound the retraining job's working set and preserve earlier events for feature construction.
    with pq.ParquetFile(input_path) as source:
        has_id = "TransactionID" in source.schema.names
    keys = ["TransactionDT", "TransactionID"] if has_id else ["TransactionDT"]
    times = pd.read_parquet(input_path, columns=keys).reset_index(drop=True)
    earlier = times.loc[times.TransactionDT < cutoff]
    if has_id and "TransactionID" in feedback:
        earlier = earlier.loc[~earlier.TransactionID.isin(feedback.TransactionID.dropna())]
    positions = earlier.sample(n=min(len(earlier), 150_000), random_state=42).index
    history = select_rows(input_path, positions)
    train = pd.concat([history, fit.drop(columns="_served_card_history")], ignore_index=True)
    features = build_features(train).reindex(columns=columns)
    labels = train.loc[features.index, "isFraud"].astype(int)
    positives = labels.sum()
    model = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, tree_method="hist",
                          random_state=42, n_jobs=2, scale_pos_weight=(len(labels) - positives) / positives)
    model.fit(features, labels)
    validation = pd.concat([
        scoring_features(record, columns, record.pop("_served_card_history"))
        for record in validate.to_dict("records")
    ])
    expected = validate.isFraud
    baseline_ap = float(average_precision_score(expected, active_model.predict_proba(validation)[:, 1]))
    candidate_ap = float(average_precision_score(expected, model.predict_proba(validation)[:, 1]))
    result = {
        "status": "candidate_ready",
        "training_feedback": len(fit),
        "validation_feedback": len(validate),
        "baseline_average_precision": baseline_ap,
        "candidate_average_precision": candidate_ap,
        "validation_gate_passed": candidate_ap >= baseline_ap,
    }
    with TemporaryDirectory(dir=candidate_dir.parent) as directory:
        staged = Path(directory)
        joblib.dump(model, staged / "xgb_model.joblib")
        (staged / "feature_columns.json").write_text(json.dumps(columns))
        (staged / "evaluation.json").write_text(json.dumps(result, indent=2))
        staged.rename(candidate_dir)
    return result


def write_status(state, payload):
    path = state / "retraining.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({**payload, "checked_at": store.now()}, indent=2))
    temporary.replace(path)
    print(json.dumps(payload), flush=True)


def monitor(input_path, min_events, min_labels, window, threshold):
    state = store.state_dir()
    state.mkdir(parents=True, exist_ok=True)
    with (state / "monitor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A monitoring run is already in progress.")
            return
        monitor_locked(input_path, state, min_events, min_labels, window, threshold)


def monitor_locked(input_path, state, min_events, min_labels, window, threshold):
    events, labels = store.monitoring_rows(window)
    if len(events) < min_events:
        write_status(state, {"status": "waiting_for_traffic", "scored_events": len(events)})
        return

    current = pd.DataFrame([flatten(event) for event in events])
    reference = pd.read_parquet(input_path, columns=["TransactionDT", *DEFAULT_COLUMNS])
    cutoff, _ = chronological_cutoffs(reference)
    reference = reference.loc[reference.TransactionDT < cutoff, DEFAULT_COLUMNS]
    summary = generate_report(reference, current, state / "drift_report.html",
                              ("Training reference", "Recent scored transactions"), threshold=threshold)
    if not summary["dataset_drift"]:
        write_status(state, {"status": "no_drift", "drift_share": summary["drift_share"]})
        return
    if len(labels) < min_labels:
        write_status(state, {"status": "waiting_for_labels", "labeled_cases": len(labels)})
        return

    batch = sha256(json.dumps(sorted((identifier, label, updated) for identifier, _, label, updated, _ in labels))
                   .encode()).hexdigest()[:16]
    candidates = state / "candidates"
    candidates.mkdir(exist_ok=True)
    candidate = candidates / batch
    if candidate.is_dir():
        result = json.loads((candidate / "evaluation.json").read_text())
    else:
        model = joblib.load("artifacts/xgb_model.joblib")
        model.set_params(n_jobs=2)
        columns = json.loads(Path("artifacts/feature_columns.json").read_text())
        result = evaluate_and_train(input_path, feedback_frame(labels), columns, model, candidate)
    if result is None:
        write_status(state, {"status": "waiting_for_label_balance", "labeled_cases": len(labels)})
    else:
        write_status(state, {**result, "candidate_id": batch})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/train.parquet")
    parser.add_argument("--min-events", type=int, default=100)
    parser.add_argument("--min-labels", type=int, default=20)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 2 <= args.min_events <= args.window or not 5 <= args.min_labels <= args.window or not 0 < args.threshold <= 1:
        parser.error("Require 2–window events, 5–window labels, and a threshold in (0, 1].")
    monitor(args.input, args.min_events, args.min_labels, args.window, args.threshold)
