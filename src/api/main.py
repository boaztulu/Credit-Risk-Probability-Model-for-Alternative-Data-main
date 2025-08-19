import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI
from src.api.pydantic_models import PredictRequest, PredictResponse

MODEL_URI = "models:/credit_scoring_model@champion"

app = FastAPI(title="Credit-Scoring API", version="0.1.0")
model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    df = pd.DataFrame([record.dict() for record in req.records])
    proba = model.predict(df)        
    preds = (proba >= 0.5).astype(int)

    return PredictResponse(
        probabilities=proba.tolist(),
        predictions=preds.tolist(),
    )
