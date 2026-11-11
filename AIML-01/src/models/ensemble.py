def blend_scores(xgb_probability: float, gnn_probability: float,
                 xgb_weight: float = 0.7) -> float:
    """Weighted score blend; tune weights on a held-out chronological set."""
    if not 0.0 <= xgb_probability <= 1.0 or not 0.0 <= gnn_probability <= 1.0:
        raise ValueError("Probabilities must be in [0, 1].")
    if not 0.0 <= xgb_weight <= 1.0:
        raise ValueError("xgb_weight must be in [0, 1].")
    return xgb_weight * xgb_probability + (1 - xgb_weight) * gnn_probability
