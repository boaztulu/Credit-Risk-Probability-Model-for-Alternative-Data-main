import numpy as np
import pandas as pd
from src.train import compute_metrics

def test_compute_metrics_shapes():
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0])
    y_prob = np.array([0.2, 0.8, 0.4, 0.1])

    m = compute_metrics(y_true, y_pred, y_prob)
    keys = {"roc_auc", "accuracy", "precision", "recall", "f1"}
    assert keys.issubset(m.keys())
    # basic sanity
    assert 0.0 <= m["roc_auc"] <= 1.0
