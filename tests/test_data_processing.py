import pandas as pd
from src.data_processing import RFMAggregator, build_preprocessing_pipeline


def _mini_df():
    data = {
        "TransactionId": [1, 2, 3, 4],
        "CustomerId":    ["A", "A", "B", "B"],
        "ChannelId":     ["web", "web", "ios", "ios"],
        "CurrencyCode":  ["USD"] * 4,
        "TransactionStartTime": pd.to_datetime(
            ["2025-06-01", "2025-06-05", "2025-06-03", "2025-06-04"]
        ),
        "Amount": [100, 150, 50, 60],
    }
    return pd.DataFrame(data)


def test_rfm_shape():
    df = _mini_df()
    agg = RFMAggregator(snapshot_date="2025-06-06").fit_transform(df)
    assert agg.shape == (2, 8)                       # ← 8 not 7
    assert {
        "CustomerId", "recency", "frequency", "monetary",
        "avg_amount", "std_amount", "ChannelId", "CurrencyCode",
    }.issubset(agg.columns)


def test_pipeline_output_dimensions():
    df = _mini_df()
    pipe = build_preprocessing_pipeline()
    X = pipe.fit_transform(df)

    assert X.shape[0] == 2

    # 5 numeric + dynamic OHE width
    ohe_cols = (
        pipe.named_steps["prep"]
            .named_transformers_["cat"]
            .named_steps["ohe"]
            .get_feature_names_out()
            .shape[0]
    )
    expected = 5 + ohe_cols
    assert X.shape[1] == expected
