from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def load_split(name: str) -> pd.DataFrame:
    tx_path = RAW / f"{name}_transaction.csv"
    id_path = RAW / f"{name}_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"Missing {tx_path}. Download the Kaggle dataset first.")
    tx = pd.read_csv(tx_path)
    if id_path.exists():
        identity = pd.read_csv(id_path)
        tx = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    return tx

def main():
    train = load_split("train")
    test = load_split("test")
    train.to_parquet(OUT / "train.parquet", index=False)
    test.to_parquet(OUT / "test.parquet", index=False)
    print(f"Prepared train={train.shape}, test={test.shape} in {OUT}")

if __name__ == "__main__":
    main()
