"""Train and serve a GNN over transactions and their historical proxy nodes."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

from src.features.proxy_history import (
    HISTORY_COLUMNS,
    PROXY_KEYS,
    chronological_cutoffs,
    proxy_stats,
)


NODE_COUNT = 1 + len(PROXY_KEYS)
NODE_FEATURES = 2 + NODE_COUNT
BATCH_SIZE = 2048


class FraudGNN(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.head(x).squeeze(-1)


def node_features(frame, history):
    nodes = np.zeros((len(frame), NODE_COUNT, NODE_FEATURES), dtype=np.float32)
    nodes[:, :, 2:] = np.eye(NODE_COUNT, dtype=np.float32)
    nodes[:, 0, 0] = np.log1p(frame["TransactionAmt"].fillna(0).clip(lower=0))
    for index, key in enumerate(PROXY_KEYS, start=1):
        counts, means = history[key]
        nodes[:, index, 0] = np.log1p(frame[key].map(means).fillna(0).clip(lower=0))
        nodes[:, index, 1] = np.log1p(frame[key].map(counts).fillna(0))
    return torch.from_numpy(nodes)


def transaction_logits(model, nodes):
    # Batch independent star graphs. Only previously observed proxy nodes send
    # messages to the transaction, so other scored transactions cannot affect it.
    rows, proxies = torch.where(nodes[:, 1:, 1] > 0)
    targets = rows * NODE_COUNT
    edges = torch.stack((targets + proxies + 1, targets))
    logits = model(nodes.flatten(0, 1), edges)
    return logits[::NODE_COUNT]


def train(input_path, output, epochs):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    frame = pd.read_parquet(input_path, columns=[*HISTORY_COLUMNS, "isFraud"])
    cutoff, validation_start = chronological_cutoffs(frame)
    history_frame = frame.loc[frame.TransactionDT < cutoff]
    history = {}
    for key in PROXY_KEYS:
        stats = proxy_stats(history_frame, key)
        history[key] = (stats.transaction_count.to_dict(), stats.mean_amount.to_dict())

    train_frame = frame.loc[(frame.TransactionDT >= cutoff) & (frame.TransactionDT < validation_start)]
    valid_frame = frame.loc[frame.TransactionDT >= validation_start]
    train_x, valid_x = (node_features(split, history) for split in (train_frame, valid_frame))
    train_y = torch.tensor(train_frame.isFraud.to_numpy(), dtype=torch.float32)
    valid_y = valid_frame.isFraud.to_numpy()
    positives = train_y.sum().item()
    if set(train_y.tolist()) != {0, 1} or set(valid_y) != {0, 1}:
        raise ValueError("Training and validation windows must each contain both fraud labels.")

    model = FraudGNN(NODE_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train_y) - positives) / positives))
    best_ap = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(train_y)).split(BATCH_SIZE):
            optimizer.zero_grad()
            loss = loss_fn(transaction_logits(model, train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            scores = torch.cat([transaction_logits(model, batch) for batch in valid_x.split(BATCH_SIZE)])
        ap = float(average_precision_score(valid_y, scores.numpy()))
        print(f"Epoch {epoch}/{epochs} | validation average precision {ap:.4f}", flush=True)
        if ap > best_ap:
            best_ap = ap
            best_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, delete=False) as temporary:
        path = Path(temporary.name)
        try:
            torch.save({
                "weights": best_weights,
                "history": history,
                "history_cutoff": cutoff,
                "validation_average_precision": best_ap,
            }, temporary)
            temporary.close()
            path.replace(output)
        finally:
            path.unlink(missing_ok=True)
    print(json.dumps({"checkpoint": str(output), "validation_average_precision": best_ap}))


class GraphScorer:
    def __init__(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model = FraudGNN(NODE_FEATURES)
        self.model.load_state_dict(checkpoint["weights"])
        self.model.eval()
        self.history = checkpoint["history"]
        self.cutoff = checkpoint["history_cutoff"]

    @torch.inference_mode()
    def score(self, payload):
        if payload["TransactionDT"] < self.cutoff:
            return None
        frame = pd.DataFrame([payload], columns=HISTORY_COLUMNS)
        nodes = node_features(frame, self.history)
        return float(torch.sigmoid(transaction_logits(self.model, nodes)).item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/train.parquet")
    parser.add_argument("--output", default="artifacts/fraud_gnn.pt")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    train(args.input, args.output, args.epochs)
