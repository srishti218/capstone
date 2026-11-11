"""Score transactions against the learned, immutable historical graph."""
import numpy as np
import pandas as pd
from scipy.special import expit


NODE_TYPES = ("account", "card", "device")


def entity_keys(frame):
    def text(column):
        return frame.get(column, pd.Series(pd.NA, index=frame.index)).astype("string").str.strip().replace("", pd.NA)

    def numeric(column):
        return frame.get(column, pd.Series(pd.NA, index=frame.index)).astype("Int64").astype("string")

    card = numeric("card1")
    address = numeric("addr1")
    return pd.DataFrame({
        "account": ("account:id:" + text("account_id")).fillna("account:proxy:" + card + ":" + address),
        "card": ("card:id:" + text("card_id")).fillna("card:" + card),
        "device": ("device:id:" + text("device_id")).fillna("device:proxy:" + text("DeviceInfo")),
    }, index=frame.index)


def node_indices(keys, indices):
    return np.column_stack([keys[kind].map(indices).fillna(0).to_numpy(dtype=np.int64) for kind in NODE_TYPES])


def ring_evidence(keys, rings):
    # A transaction connecting multiple known components gets one deterministic lead.
    account, card, device = (keys[kind].map(rings).astype("string") for kind in NODE_TYPES)
    return account.fillna(card).fillna(device)


class RingScorer:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as artifact:
            keys = artifact["keys"].tolist()
            self.indices = dict(zip(keys, artifact["node_ids"].tolist()))
            self.rings = {key: ring for key, ring in zip(keys, artifact["ring_ids"].tolist()) if ring}
            self.cutoff = int(artifact["cutoff"])
            self.calibrators = artifact["calibrators"]
            self.embeddings = artifact["embeddings"]
            self.w1, self.b1, self.w2, self.b2 = (artifact[key] for key in ("w1", "b1", "w2", "b2"))

    def score(self, payload):
        if payload["TransactionDT"] < self.cutoff:
            return {}
        keys = entity_keys(pd.DataFrame([payload]))
        ids = node_indices(keys, self.indices)
        if not ids.any():
            return {}
        amount = np.float32(np.log1p(payload["TransactionAmt"]))
        features = np.append(self.embeddings[ids].ravel(), amount)
        # The snapshot is fixed until retraining, so serving only needs the learned head.
        hidden = np.maximum(0, self.w1 @ features + self.b1)
        logits = self.w2 @ hidden + self.b2
        fraud, ring = expit(logits * self.calibrators[:, 0] + self.calibrators[:, 1]).tolist()
        ring_id = ring_evidence(keys, self.rings).iloc[0]
        return {"gnn_score": fraud, "ring_score": ring if pd.notna(ring_id) else None,
                "ring_id": ring_id if pd.notna(ring_id) else None}
