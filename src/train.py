from pathlib import Path
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

MLFLOW_TRACKING_URI = "mlruns"   # local folder inside repo


# ------------------------------------------------------------------ #
# helper: score dict                                                 #
# ------------------------------------------------------------------ #
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def main(cli_args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Parquet with features + is_high_risk")
    p.add_argument("--model-name", default="credit_scoring_model")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(cli_args)

    # ------------------------------------------------------------------ #
    # data load & split                                                  #
    # ------------------------------------------------------------------ #
    df = pd.read_parquet(Path(args.data))
    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # ------------------------------------------------------------------ #
    # candidate 1 – Logistic Regression (liblinear)                      #
    # ------------------------------------------------------------------ #
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    log_grid = {"C": [0.01, 0.1, 1, 10]}

    gs_lr = GridSearchCV(
        log_reg, log_grid, scoring="roc_auc", cv=5, n_jobs=-1, refit=True
    ).fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # candidate 2 – Gradient Boosting                                    #
    # ------------------------------------------------------------------ #
    gb = GradientBoostingClassifier(random_state=args.seed)
    gb_grid = {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}

    gs_gb = GridSearchCV(
        gb, gb_grid, scoring="roc_auc", cv=5, n_jobs=-1, refit=True
    ).fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # evaluate on hold-out set                                           #
    # ------------------------------------------------------------------ #
    runs = [("logreg", gs_lr), ("gb", gs_gb)]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    best_run_id = None
    best_auc = -1.0

    for tag, gs in runs:
        y_pred = gs.best_estimator_.predict(X_test)
        y_prob = gs.best_estimator_.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)

        with mlflow.start_run(run_name=tag) as run:
            mlflow.log_params(gs.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(gs.best_estimator_, "model")

            if metrics["roc_auc"] > best_auc:
                best_auc = metrics["roc_auc"]
                best_run_id = run.info.run_id

    # ------------------------------------------------------------------ #
    # register the champion                                             #
    # ------------------------------------------------------------------ #
    client = mlflow.tracking.MlflowClient()
    best_model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(best_model_uri, args.model_name)
    client.transition_model_version_stage(
        name=args.model_name, version=mv.version, stage="Staging", archive_existing_versions=True
    )

    print(f"✓ Registered {args.model_name} v{mv.version} (ROC-AUC={best_auc:.3f})")


if __name__ == "__main__":
    main()
