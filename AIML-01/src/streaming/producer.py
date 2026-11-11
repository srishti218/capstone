"""Replay prepared historical transactions without leaking their fraud labels."""
import argparse
import time

import pandas as pd
from kafka.errors import KafkaError

from src.data.parquet_rows import select_rows
from src.streaming.consumer import TRANSACTIONS, publisher


EVENT_COLUMNS = ("TransactionAmt", "TransactionDT", "card1", "addr1", "DeviceInfo",
                 "P_emaildomain", "account_id", "card_id", "device_id")


def transactions(input_path, start_time, limit):
    is_parquet = str(input_path).lower().endswith(".parquet")
    frame = pd.read_parquet(input_path, columns=["TransactionDT"]) if is_parquet else pd.read_csv(input_path)
    frame = frame.reset_index(drop=True)
    times = frame.TransactionDT.sort_values(kind="stable")
    cutoff = times.quantile(0.8) if start_time is None else start_time
    positions = times[times >= cutoff].head(limit).index
    window = select_rows(input_path, positions).sort_values("TransactionDT", kind="stable") if is_parquet and len(positions) else frame.iloc[positions]
    for row in window.to_dict(orient="records"):
        payload = {key: value for key, value in row.items() if pd.notna(value)}
        transaction_id = f"replay-{int(payload.pop('TransactionID'))}"
        event = {key: payload.pop(key) for key in EVENT_COLUMNS if key in payload}
        for key in ("account_id", "card_id", "device_id"):
            if key in event:
                event[key] = str(event[key])
        event.update(transaction_id=transaction_id, extra_features={
            key: value for key, value in payload.items() if key not in {"isFraud", "ring_label"}
        })
        yield event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/train.parquet")
    parser.add_argument("--start-time", type=int, help="First TransactionDT to replay (default: last 20%%).")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--interval", type=float, default=0.05)
    args = parser.parse_args()
    if args.limit < 1 or args.interval < 0:
        parser.error("--limit must be positive and --interval must be nonnegative.")

    for attempt in range(30):
        try:
            producer = publisher()
            break
        except KafkaError:
            if attempt == 29:
                raise
            time.sleep(2)
    count = 0
    try:
        for event in transactions(args.input, args.start_time, args.limit):
            producer.send(TRANSACTIONS, key=event["transaction_id"].encode(), value=event).get(timeout=20)
            count += 1
            time.sleep(args.interval)
    finally:
        producer.close(timeout=5)
    print(f"Replayed {count} transactions.")


if __name__ == "__main__":
    main()
