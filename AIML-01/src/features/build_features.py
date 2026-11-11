import numpy as np
import pandas as pd

TARGET = "isFraud"
DROP_COLS = {"TransactionID", "account_id", "card_id", "device_id", "ring_label", TARGET}

def build_features(df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
    """Create leakage-aware historical features, sorted by event time.

    card1 is only a proxy grouping key in IEEE-CIS, not a canonical user ID.
    """
    x = df.copy()
    if "TransactionDT" in x:
        x = x.sort_values("TransactionDT", kind="stable")

    # Amount relative to prior history for the card1 proxy.
    if "card1" in x and "TransactionAmt" in x:
        grp = x.groupby("card1", dropna=False)["TransactionAmt"]
        prior_mean = grp.transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        prior_count = grp.transform(lambda s: s.shift(1).expanding(min_periods=1).count())
        x["amount_vs_card1_prior_mean"] = (
            x["TransactionAmt"] / prior_mean.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        x["card1_prior_tx_count"] = prior_count.fillna(0)

        prior_std = grp.transform(lambda s: s.shift(1).expanding(min_periods=2).std())
        x["amount_zscore_vs_card1"] = (
            (x["TransactionAmt"] - prior_mean) / prior_std.replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Previous observed device/address and whether they changed.
    for key in ("DeviceInfo", "addr1", "P_emaildomain"):
        if key in x:
            prev = x.groupby("card1", dropna=False)[key].shift(1) if "card1" in x else x[key].shift(1)
            x[f"{key}_changed_from_previous"] = (
                x[key].notna() & prev.notna() & x[key].ne(prev)
            ).astype("int8")

    # Simple velocity proxy: number of prior card1 transactions in the dataset order.
    if "card1" in x:
        x["card1_velocity_count"] = x.groupby("card1", dropna=False).cumcount()

    # Timestamp-derived features.
    if "TransactionDT" in x:
        x["transaction_hour_proxy"] = ((x["TransactionDT"] // 3600) % 24).astype("int16")
        x["transaction_day_proxy"] = (x["TransactionDT"] // 86400).astype("int32")

    # Drop high-cardinality raw text fields only after deriving change indicators.
    for col in ("DeviceInfo", "id_30", "id_31", "id_33"):
        if col in x:
            x = x.drop(columns=col)

    # Remove IDs and target from model inputs.
    x = x.drop(columns=[c for c in DROP_COLS if c in x], errors="ignore")
    # Convert object/category columns to stable category codes.
    for col in x.select_dtypes(include=["object", "category"]).columns:
        x[col] = x[col].astype("category").cat.codes.replace(-1, np.nan)
    x = x.replace([np.inf, -np.inf], np.nan)
    return x

def align_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=columns)
