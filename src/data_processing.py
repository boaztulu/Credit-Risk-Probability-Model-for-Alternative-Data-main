from pathlib import Path
import argparse
import pandas as pd
from packaging import version
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# --------------------------------------------------------------------------- #
# 0.  Helper: version-safe One-Hot encoder                                    #
# --------------------------------------------------------------------------- #
def safe_ohe() -> OneHotEncoder:
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


# --------------------------------------------------------------------------- #
# 1.  RFM aggregator                                                          #
# --------------------------------------------------------------------------- #
class RFMAggregator(BaseEstimator, TransformerMixin):
    """Collapse transactions to one row per CustomerId with R, F, M stats."""

    def __init__(self, snapshot_date: str | None = None):
        self.snapshot_date = snapshot_date  # "YYYY-MM-DD" or None

    def fit(self, X: pd.DataFrame, y=None):
        self.snapshot_ = (
            pd.to_datetime(self.snapshot_date)
            if self.snapshot_date
            else X["TransactionStartTime"].max() + pd.Timedelta(days=1)
        )
        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()

        grp = (
            df.groupby("CustomerId")
            .agg(
                recency=("TransactionStartTime",
                         lambda x: (self.snapshot_ - x.max()).days),
                frequency=("TransactionId", "count"),
                monetary=("Amount", "sum"),
                avg_amount=("Amount", "mean"),
                std_amount=("Amount", "std"),
            )
            .fillna(0)      # std_amount NaN → 0 when single txn
            .reset_index()
        )

        latest_cat = (
            df.sort_values("TransactionStartTime")
              .groupby("CustomerId")
              .tail(1)[["CustomerId", "ChannelId", "CurrencyCode"]]
        )

        return pd.merge(grp, latest_cat, on="CustomerId", how="left")


# --------------------------------------------------------------------------- #
# 2.  ColumnTransformer pipeline                                              #
# --------------------------------------------------------------------------- #
NUM_COLS = ["recency", "frequency", "monetary", "avg_amount", "std_amount"]
CAT_COLS = ["ChannelId", "CurrencyCode"]


def build_preprocessing_pipeline() -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", safe_ohe()),
        ]
    )

    col_tf = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUM_COLS),
            ("cat", categorical_pipe, CAT_COLS),
        ]
    )

    return Pipeline(
        steps=[
            ("rfm", RFMAggregator()),
            ("prep", col_tf),
        ]
    )


# --------------------------------------------------------------------------- #
# 3.  CLI entry-point                                                         #
# --------------------------------------------------------------------------- #
def main(cli_args=None):
    parser = argparse.ArgumentParser(description="Run feature engineering")
    parser.add_argument("--raw", required=True, help="Path to raw CSV file")
    parser.add_argument("--out", required=True, help="Path to output Parquet")
    parser.add_argument(
        "--snapshot",
        help="Override snapshot date YYYY-MM-DD (default = max(date)+1)",
    )
    args = parser.parse_args(cli_args or None)

    raw_path = Path(args.raw)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(raw_path, parse_dates=["TransactionStartTime"])

    pipe = build_preprocessing_pipeline()
    X_proc = pipe.fit_transform(df_raw)

    feature_names = pipe.named_steps["prep"].get_feature_names_out()
    pd.DataFrame(X_proc, columns=feature_names).to_parquet(out_path, index=False)

    print(f"✓ Saved processed features → {out_path.resolve()}")


if __name__ == "__main__":
    main()
