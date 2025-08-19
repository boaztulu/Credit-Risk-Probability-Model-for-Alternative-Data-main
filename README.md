# Credit-Risk Probability Model for Alternative Data

> **Bati Bank × e-Commerce BNPL pilot**  
> Data science pipeline that transforms raw transaction logs into
> customer-level risk probabilities, registers the champion model in MLflow,
> and serves real-time predictions via FastAPI + Docker.

---

## 1 Business Context — “Credit Scoring Business Understanding”

| ☑                                    | Basel II takeaway                                                 | Impact on this repo                                                                                                                           |
| ------------------------------------ | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Interpretability & documentation** | Institutions must justify risk estimates and keep an audit trail. | Pipeline is decomposed into _EDA → feature scripts → MLflow runs → versioned API_. MLflow artifacts + signatures create a reproducible trail. |
| **No labelled defaults**             | We lack real “default = 1” tags in the e-commerce dataset.        | We engineered an RFM-based **proxy label** `is_high_risk` (Task 4).                                                                           |
| **Model risk vs. performance**       | Simpler scorecards are easier to defend, GBMs lift AUC.           | We train **LogReg + GB**; MLflow compares ROC-AUC, registers the winner and keeps the full run history.                                       |

---

## 2 Exploratory Data Analysis (Notebook `notebooks/1.0-eda.ipynb`)

| Insight                                                                                         | Proof                          |
| ----------------------------------------------------------------------------------------------- | ------------------------------ |
| **Spend is heavily right-skewed** — 80 % of customers spend \< \$250 total.                     | log-histogram & summary stats. |
| **Three dominant channels** (`web`, `ios`, `android`) cover 94 % of traffic.                    | pie chart.                     |
| **`FraudResult = 1` transactions are rare ( \< 0.3 %)** and nearly all are voided the same day. | grouped bar.                   |
| **Recency & Monetary inversely correlated (ρ ≈ –0.58)** – fresh customers spend more.           | corr-heatmap.                  |

---

## 3 Feature Engineering (Task 3)

- **`RFMAggregator`** collapses raw txns → one row per `CustomerId`  
  `recency | frequency | monetary | avg_amount | std_amount | ChannelId | CurrencyCode`
- **Pre-processing pipeline (`build_preprocessing_pipeline`)**
  - numeric → median-impute → StandardScaler
  - categorical → mode-impute → One-Hot (dense)
  - outputs a NumPy matrix ready for models.

Unit tests (`tests/test_data_processing.py`) guarantee shape & transform.

---

## 4 Proxy Target Engineering (Task 4)

1. Calculate R F M per customer.
2. **K-Means (k = 3)** on standardised RFM.
3. Cluster with lowest _F_ & _M_ = **`is_high_risk = 1`**.
4. 19 .7 % of customers fall in this bucket → reasonable class balance.

Test `tests/test_target_engineering.py` checks the worst toy-customer is flagged.

---

## 5 Model Training & Tracking (Task 5)

| Step           | Details                                                                                                                                           |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data**       | `features_with_target.parquet` (numeric only + label)                                                                                             |
| **Algorithms** | LogisticRegression (liblinear, balanced) • GradientBoostingClassifier                                                                             |
| **Tuning**     | GridSearchCV (AUC, 5-fold)                                                                                                                        |
| **Metrics**    | ROC-AUC, accuracy, precision, recall, F1                                                                                                          |
| **MLflow**     | Each grid-search run logs params, metrics, model + signature & example. Best run registered as **alias `champion`** under `credit_scoring_model`. |

`src/train.py` can also accept separate `--features` and `--labels` files.

---

## 6 Model Serving (Task 6)

```
graph LR
  browser--REST-->FastAPI
  FastAPI--load@startup-->MLflow[MLflow model registry]
  FastAPI--predict-->Model
```

FastAPI app (src/api/main.py) loads alias models:/credit_scoring_model@champion.

Pydantic schemas (src/api/pydantic_models.py) validate request / response.

Docker

```
docker compose up --build
# Swagger UI → http://localhost:8000/docs
```

CI/CD (.github/workflows/ci.yml)
flake8 → pytest → docker-build.

## 7 Project Structure

```
credit-risk-model/
├── data/
│   ├── raw/                 # original CSV
│   └── processed/           # parquet artefacts
├── notebooks/1.0-eda.ipynb
├── src/
│   ├── data_processing.py
│   ├── target_engineering.py
│   ├── train.py
│   ├── api/
│   │   ├── main.py
│   │   └── pydantic_models.py
│   └── __init__.py
├── tests/                   # pytest unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .flake8
└── README.md
```

## 8 Quick-Start (local)

```
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Feature engineering (Task 3 output)
python -m src.data_processing \
       --raw data/raw/transactions.csv \
       --out data/processed/features.parquet

# 3. Proxy labels
python -m src.target_engineering \
       --raw data/raw/transactions.csv \
       --out data/processed/high_risk_labels.parquet

# 4. Merge & train
python - <<'PY'
import pandas as pd, pathlib
root = pathlib.Path('.')
feat = pd.read_parquet(root/"data/processed/features.parquet")
lab  = pd.read_parquet(root/"data/processed/high_risk_labels.parquet")
feat["is_high_risk"] = lab["is_high_risk"].values
feat.to_parquet(root/"data/processed/features_with_target.parquet", index=False)
PY

python -m src.train --features data/processed/features_with_target.parquet

# 5. Serve
docker compose up --build
# → POST /predict with JSON matching Pydantic schema
```
