import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.features.proxy_history import chronological_cutoffs


DEFAULT_COLUMNS = ["TransactionAmt", "dist1", "C1", "C2", "D1", "D4"]
SAMPLE_SIZE = 10_000


def read_window(path):
    return pd.read_parquet(path) if Path(path).suffix.lower() == ".parquet" else pd.read_csv(path)


def generate_report(reference, current, output, windows, threshold=0.5):
    columns = [
        name for name in reference.columns
        if name in current and reference[name].notna().any() and current[name].notna().any()
    ]
    if not columns:
        raise ValueError("Reference and current windows need comparable columns with observed values.")
    reference = reference[columns].sample(n=min(len(reference), SAMPLE_SIZE), random_state=42)
    current = current[columns].sample(n=min(len(current), SAMPLE_SIZE), random_state=42)
    report = Report([DataDriftPreset(drift_share=threshold)], include_tests=False).run(
        current_data=current, reference_data=reference
    )
    metrics = report.dict()["metrics"]
    drift = next(item for item in metrics if item["config"]["type"].endswith(":DriftedColumnsCount"))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_window": windows[0],
        "current_window": windows[1],
        "reference_rows": len(reference),
        "current_rows": len(current),
        "drifted_columns": int(drift["value"]["count"]),
        "compared_columns": len(columns),
        "drift_share": float(drift["value"]["share"]),
        "dataset_drift": bool(drift["value"]["share"] >= drift["config"]["drift_share"]),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        html = Path(directory) / "report.html"
        metadata = html.with_suffix(".json")
        report.save_html(str(html))
        metadata.write_text(json.dumps(summary, indent=2))
        html.replace(output)
        metadata.replace(output.with_suffix(".json"))
    print(json.dumps(summary, indent=2))
    print(f"Saved drift report: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare reference and current transaction windows.")
    parser.add_argument("--reference", help="Reference CSV or Parquet file")
    parser.add_argument("--current", help="Current CSV or Parquet file")
    parser.add_argument("--input", default="data/processed/train.parquet",
                        help="Prepared training Parquet used when no windows are supplied")
    parser.add_argument("--output", default="artifacts/drift_report.html")
    args = parser.parse_args()
    if bool(args.reference) != bool(args.current):
        parser.error("provide both --reference and --current, or omit both")
    if Path(args.output).suffix.lower() != ".html":
        parser.error("--output must be an HTML filename")
    if args.reference:
        reference, current = (read_window(path) for path in (args.reference, args.current))
        windows = ("Provided reference data", "Provided current data")
    else:
        frame = pd.read_parquet(args.input, columns=["TransactionDT", *DEFAULT_COLUMNS])
        cutoff, validation_start = chronological_cutoffs(frame)
        reference = frame.loc[frame.TransactionDT < cutoff, DEFAULT_COLUMNS]
        current = frame.loc[frame.TransactionDT >= validation_start, DEFAULT_COLUMNS]
        windows = (f"TransactionDT < {cutoff}", f"TransactionDT ≥ {validation_start}")
    generate_report(reference, current, args.output, windows)


if __name__ == "__main__":
    main()
