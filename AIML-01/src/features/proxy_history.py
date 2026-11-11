import pandas as pd


PROXY_KEYS = ("card1", "addr1", "DeviceInfo", "P_emaildomain")
HISTORY_COLUMNS = ["TransactionDT", "TransactionAmt", *PROXY_KEYS]


def chronological_cutoffs(frame):
    times = frame["TransactionDT"].sort_values()
    if times.empty or times.isna().any():
        raise ValueError("Expected nonempty transactions with a TransactionDT for every row.")
    history_end = int(times.iloc[len(times) * 3 // 5])
    validation_start = int(times.iloc[len(times) * 4 // 5])
    if not times.iloc[0] < history_end < validation_start:
        raise ValueError("Expected three nonempty chronological windows.")
    return history_end, validation_start


def proxy_stats(history, key):
    return history.groupby(key)["TransactionAmt"].agg(
        transaction_count="size", mean_amount="mean"
    ).dropna(subset=["mean_amount"])
