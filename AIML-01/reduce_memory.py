import pandas as pd

train = pd.read_csv(
    "./data/raw/train_transaction.csv",
    nrows=50000
)

print(train.shape)
print(
    train.memory_usage(deep=True).sum() / 1024**2,
    "MB"
)