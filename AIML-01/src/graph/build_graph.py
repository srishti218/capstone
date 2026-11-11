"""Build a simple heterogeneous-proxy graph edge list from IEEE-CIS rows."""
import pandas as pd

def build_edges(df: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for col in ("card1", "DeviceInfo", "addr1", "P_emaildomain"):
        if col not in df:
            continue
        temp = df[["TransactionID", col]].dropna()
        for value, group in temp.groupby(col):
            ids = group["TransactionID"].tolist()
            if len(ids) > 1:
                # Star edges avoid a quadratic clique for frequently shared values.
                anchor = ids[0]
                pairs.extend((anchor, other, col) for other in ids[1:])
    return pd.DataFrame(pairs, columns=["src_transaction", "dst_transaction", "shared_key_type"])
