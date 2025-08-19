import pandas as pd
from src.target_engineering import label_high_risk, make_rfm

def _toy_txn():
    data = {
        "TransactionId": [1, 2, 3, 4, 5, 6],
        "CustomerId":    ["C0", "C1", "C1", "C2", "C2", "C2"],
        "ChannelId":     ["web", "web", "web", "ios", "ios", "ios"],
        "CurrencyCode":  ["USD"] * 6,
        "TransactionStartTime": pd.to_datetime(
            ["2025-06-01", "2025-06-02", "2025-06-05",
             "2025-06-03", "2025-06-04", "2025-06-04"]
        ),
        "Amount": [50, 400, 400, 300, 300, 300],
    }
    return pd.DataFrame(data)

def test_label_high_risk():
    df_txn = _toy_txn()
    rfm = make_rfm(df_txn, snapshot="2025-06-06")
    y = label_high_risk(rfm, random_state=0)

    # one and only one high-risk label
    assert y.sum() == 1
    worst_idx = rfm.index[rfm["CustomerId"] == "C0"][0]
    assert y.loc[worst_idx] == 1
