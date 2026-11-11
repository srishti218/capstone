import pandas as pd
from src.features.build_features import build_features

def test_feature_builder_creates_historical_features():
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [1, 2, 3],
        "TransactionAmt": [10.0, 20.0, 30.0],
        "card1": [7, 7, 7],
        "DeviceInfo": ["A", "A", "B"]
    })
    result = build_features(df)
    assert "amount_vs_card1_prior_mean" in result
    assert "DeviceInfo_changed_from_previous" in result
    assert result.iloc[2]["DeviceInfo_changed_from_previous"] == 1
