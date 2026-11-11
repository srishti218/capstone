"""Chronological account–card–device graph training and online inference."""
import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from src.features.proxy_history import chronological_cutoffs
from src.graph.scoring import NODE_TYPES, entity_keys, node_indices, ring_evidence


REQUIRED = ["TransactionDT", "TransactionAmt", "isFraud"]
OPTIONAL = ["account_id", "device_id", "card_id", "card1", "addr1", "DeviceInfo", "ring_label"]
MAX_DEGREE = 40
MAX_RING_ACCOUNTS = 100


def build_entity_graph(history):
    keys = entity_keys(history)
    indices = {}
    features = [[0.0] * 6]  # Zero is reserved for missing or unseen entities.
    for type_index, kind in enumerate(NODE_TYPES):
        grouped = history["TransactionAmt"].groupby(keys[kind]).agg(["size", "mean"])
        for key, (count, mean) in grouped.iterrows():
            indices[key] = len(features)
            identity = [float(index == type_index) for index in range(len(NODE_TYPES))]
            features.append([np.log1p(count), np.log1p(max(0, mean)), 0.0, *identity])

    ids = node_indices(keys, indices)
    pairs = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        edges = ids[:, [left, right]]
        pairs.append(np.unique(edges[(edges != 0).all(axis=1)], axis=0))
    edges = np.concatenate(pairs)
    degrees = np.bincount(edges.ravel(), minlength=len(features))
    # Broad proxy hubs (for example a generic device name) should not join unrelated accounts.
    edges = edges[(degrees[edges] <= MAX_DEGREE).all(axis=1)]
    directed = np.concatenate((edges, edges[:, ::-1])).T.copy()
    adjacency = coo_matrix(
        (np.ones(directed.shape[1]), tuple(directed)), shape=(len(features), len(features))
    ).tocsr()
    _, components = connected_components(adjacency, directed=False)
    unique_accounts = np.unique(ids[:, 0])
    unique_accounts = unique_accounts[unique_accounts != 0]
    accounts = np.bincount(components[unique_accounts], minlength=len(features))
    candidates = np.flatnonzero((accounts >= 2) & (accounts <= MAX_RING_ACCOUNTS))
    component_ids = {}
    for key, index in sorted(indices.items()):
        component_ids.setdefault(int(components[index]), sha256(key.encode()).hexdigest()[:16])
    eligible = {int(component): component_ids[int(component)] for component in candidates}
    rings = {key: eligible[int(components[index])] for key, index in indices.items()
             if int(components[index]) in eligible}

    node_features = np.asarray(features, dtype=np.float32)
    node_features[:, 2] = np.log1p(np.diff(adjacency.indptr))
    return indices, rings, torch.from_numpy(node_features), torch.from_numpy(directed)


def training_rows(frame, indices, rings):
    keys = entity_keys(frame)
    fraud = frame.isFraud.to_numpy(dtype=np.float32)
    if "ring_label" in frame:
        ring = frame["ring_label"].to_numpy(dtype=np.float32)
        if not np.isin(ring, [0, 1]).all():
            raise ValueError("ring_label must contain only 0 and 1.")
    else:
        evidence = pd.DataFrame({"ring": ring_evidence(keys, rings), "account": keys["account"]})
        suspicious = evidence.loc[fraud == 1].groupby("ring").account.nunique()
        ring = ((fraud == 1) & (evidence.ring.map(suspicious).fillna(0) >= 2)).to_numpy(dtype=np.float32)
    amount = np.log1p(frame["TransactionAmt"].fillna(0).clip(lower=0).to_numpy(dtype=np.float32))
    return (
        torch.from_numpy(node_indices(keys, indices)),
        torch.from_numpy(amount[:, None]),
        torch.from_numpy(np.column_stack((fraud, ring))),
    )


class RingGNN(nn.Module):
    def __init__(self, width=24):
        super().__init__()
        self.first = SAGEConv(6, width)
        self.second = SAGEConv(width, width)
        self.head = nn.Sequential(nn.Linear(width * len(NODE_TYPES) + 1, width), nn.ReLU(), nn.Linear(width, 2))

    def encode(self, features, edges):
        embeddings = self.second(F.relu(self.first(features, edges)), edges)
        # The reserved zero node represents absent information.
        return torch.cat((torch.zeros_like(embeddings[:1]), embeddings[1:]))

    def forward(self, embeddings, ids, amounts):
        return self.head(torch.cat((embeddings[ids].flatten(1), amounts), dim=1))


def train(input_path, output, epochs):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    columns = pq.ParquetFile(input_path).schema.names
    data = pd.read_parquet(input_path, columns=REQUIRED + [key for key in OPTIONAL if key in columns])
    cutoff, validation_start = chronological_cutoffs(data)
    holdout = data.loc[data.TransactionDT >= validation_start, "TransactionDT"].sort_values()
    test_start = int(holdout.iloc[len(holdout) // 2])
    if test_start <= validation_start:
        raise ValueError("Expected disjoint calibration and evaluation windows.")
    indices, rings, features, edges = build_entity_graph(data.loc[data.TransactionDT < cutoff])
    training = training_rows(data.loc[(data.TransactionDT >= cutoff) & (data.TransactionDT < validation_start)], indices, rings)
    calibration = training_rows(data.loc[(data.TransactionDT >= validation_start) & (data.TransactionDT < test_start)], indices, rings)
    evaluation = training_rows(data.loc[data.TransactionDT >= test_start], indices, rings)
    if any(set(rows[2][:, 0].tolist()) != {0.0, 1.0} for rows in (training, calibration, evaluation)):
        raise ValueError("Training, calibration and evaluation each require both fraud classes.")
    if training[2][:, 1].sum() == 0:
        raise ValueError("No positive ring examples. Supply ring_label or connected fraud examples.")

    model = RingGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
    positives = training[2].sum(0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(len(training[2]) - positives) / positives)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(model.encode(features, edges), *training[:2])
        loss = loss_fn(predictions, training[2])
        loss.backward()
        optimizer.step()
        print(f"Graph epoch {epoch + 1}/{epochs} loss={loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(features, edges)
        calibration_logits = model(embeddings, *calibration[:2]).numpy()
        evaluation_logits = model(embeddings, *evaluation[:2]).numpy()
    calibrators = []
    metrics = {}
    for index, target in enumerate(("fraud", "ring")):
        labels = calibration[2][:, index].numpy()
        if len(np.unique(labels)) != 2:
            raise ValueError(f"Calibration requires both {target} classes.")
        calibrator = LogisticRegression().fit(calibration_logits[:, [index]], labels)
        calibrators.append([float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])])
        metrics[f"{target}_average_precision"] = float(average_precision_score(
            evaluation[2][:, index].numpy(), calibrator.predict_proba(evaluation_logits[:, [index]])[:, 1]
        ))

    artifact = {
        "weights": model.state_dict(), "indices": indices, "rings": rings, "features": features,
        "edges": edges, "embeddings": embeddings, "cutoff": cutoff, "calibrators": calibrators, "metrics": metrics,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        serving = checkpoint.with_suffix(".npz")
        torch.save(artifact, checkpoint)
        head = artifact["weights"]
        np.savez_compressed(
            serving, keys=list(indices), node_ids=list(indices.values()),
            ring_ids=[rings.get(key, "") for key in indices], cutoff=cutoff, calibrators=calibrators,
            embeddings=embeddings.numpy(), w1=head["head.0.weight"].numpy(), b1=head["head.0.bias"].numpy(),
            w2=head["head.2.weight"].numpy(), b2=head["head.2.bias"].numpy(),
        )
        checkpoint.replace(output)
        serving.replace(output.with_suffix(".npz"))
    print(f"Saved {output}: {len(indices):,} entities, {edges.shape[1] // 2:,} links, "
          f"{len(set(rings.values())):,} candidate components. {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/train.parquet")
    parser.add_argument("--output", default="artifacts/fraud_ring_gnn.pt")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    train(args.input, args.output, args.epochs)
