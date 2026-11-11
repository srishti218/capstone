"""Feature construction shared by the scoring endpoint and candidate evaluation."""
import pandas as pd

from src.features.build_features import build_features


def scoring_features(payload, columns, card_history=None):
    frame = build_features(pd.DataFrame([payload]))
    if card_history is not None:
        count = card_history["transaction_count"]
        mean = card_history["mean_amount"]
        frame["amount_vs_card1_prior_mean"] = payload["TransactionAmt"] / mean if mean else 1.0
        frame["card1_prior_tx_count"] = count
        frame["card1_velocity_count"] = count
    return frame.reindex(columns=columns)
