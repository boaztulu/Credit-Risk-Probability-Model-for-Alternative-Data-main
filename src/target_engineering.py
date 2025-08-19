from pathlib import Path
import argparse
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.data_processing import RFMAggregator  # <-- reuse existing class

RFM_COLS = ["recency", "frequency", "monetary"]


def make_rfm(df_txn: pd.DataFrame, snapshot: str | None = None) -> pd.DataFrame:
    agg = RFMAggregator(snapshot_date=snapshot)
    return agg.fit_transform(df_txn)  # includes ChannelId & CurrencyCode


def label_high_risk(rfm: pd.DataFrame, random_state=42) -> pd.Series:
    rfm_scaled = StandardScaler().fit_transform(rfm[RFM_COLS])

    km = KMeans(n_clusters=3, random_state=random_state, n_init="auto")
    clusters = km.fit_predict(rfm_scaled)

    # pick the cluster with lowest F & M (and highest R) as high risk
    profile = (
        rfm.assign(cluster=clusters)
           .groupby("cluster")[RFM_COLS]
           .mean()
           .sort_values(["frequency", "monetary"], ascending=[True, True])
    )
    high_risk_cluster = profile.index[0]
    return pd.Series((clusters == high_risk_cluster).astype(int), index=rfm.index, 
    name='is_high_risk')


def main(cli_args=None):
    p = argparse.ArgumentParser(description="Create proxy high-risk labels")
    p.add_argument("--raw", required=True, help="Raw transactions CSV")
    p.add_argument("--out", required=True, help="Output parquet (CustomerId + label)")
    p.add_argument("--snapshot", help="Snapshot date YYYY-MM-DD (default max+1)")
    p.add_argument("--seed", type=int, default=42, help="K-Means random_state")
    args = p.parse_args(cli_args)

    df_raw = pd.read_csv(Path(args.raw), parse_dates=["TransactionStartTime"])

    rfm = make_rfm(df_raw, snapshot=args.snapshot)
    rfm["is_high_risk"] = label_high_risk(rfm, random_state=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rfm[["CustomerId", "is_high_risk"]].to_parquet(out_path, index=False)
    print(f"✓ Saved high-risk labels → {out_path.resolve()}")


if __name__ == "__main__":
    main()
