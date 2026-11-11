import argparse
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

from src.features.proxy_history import chronological_cutoffs, proxy_stats


REPO = Path(__file__).resolve().parents[2] / "feast"
FEATURE_NAMES = ("transaction_count", "mean_amount", "history_cutoff")
VIEW = "card_history"


def materialize(input_path):
    frame = pd.read_parquet(input_path, columns=["card1", "TransactionAmt", "TransactionDT"])
    cutoff, _ = chronological_cutoffs(frame)
    features = proxy_stats(frame.loc[frame.TransactionDT < cutoff], "card1").reset_index()
    if features.empty:
        raise ValueError("No card history is available in the reference window.")
    features["card1"] = features["card1"].astype("int64")
    features["history_cutoff"] = cutoff
    # IEEE-CIS times are relative seconds. Use a fixed origin for Feast's timestamps.
    timestamp = pd.Timestamp("2000-01-01", tz="UTC") + pd.Timedelta(seconds=cutoff)
    features["event_timestamp"] = timestamp
    entity = Entity(name="card_proxy", join_keys=["card1"], value_type=ValueType.INT64)
    view = FeatureView(
        name=VIEW,
        entities=[entity],
        ttl=None,
        schema=[Field(name=name, dtype=dtype) for name, dtype in zip(FEATURE_NAMES, (Int64, Float64, Int64))],
        source=FileSource(
            name="card_history_source", path="data/card_history.parquet", timestamp_field="event_timestamp"
        ),
    )
    destination = REPO / "data"
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination, prefix=".snapshot-") as directory:
        staged = Path(directory) / "data"
        staged.mkdir()
        features.to_parquet(staged / "card_history.parquet", index=False)
        store = FeatureStore(repo_path=directory, fs_yaml_file=REPO / "feature_store.yaml")
        store.apply([entity, view])
        store.materialize(timestamp - timedelta(seconds=1), timestamp + timedelta(seconds=1), [VIEW])
        # Publish the online database last so readers only see complete snapshots.
        for name in ("card_history.parquet", "registry.db", "online_store.db"):
            (staged / name).replace(destination / name)
    print(f"Materialized history for {len(features):,} card proxies through TransactionDT < {cutoff}.")


def lookup(store, card, transaction_time):
    row = store.get_online_features(
        features=[f"{VIEW}:{name}" for name in FEATURE_NAMES], entity_rows=[{"card1": card}]
    ).to_dict()
    if row["history_cutoff"][0] is None or transaction_time < row["history_cutoff"][0]:
        return None
    return {name: row[name][0] for name in FEATURE_NAMES[:2]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and materialize the local Feast card snapshot.")
    parser.add_argument("--input", default="data/processed/train.parquet")
    materialize(parser.parse_args().input)
