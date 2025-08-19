from pydantic import BaseModel, Field
from typing import List


class FeatureVector(BaseModel):
    # replace … with the actual feature column names from features.parquet
    num__recency: float = Field(..., example=3.0)
    num__frequency: float = Field(..., example=12.0)
    num__monetary: float = Field(..., example=450.0)
    num__avg_amount: float = Field(..., example=37.5)
    num__std_amount: float = Field(..., example=5.6)
    cat__ChannelId_ios: int = Field(..., example=0)
    cat__ChannelId_web: int = Field(..., example=1)
    cat__CurrencyCode_USD: int = Field(..., example=1)
    # add / remove one-hot columns to match your pipeline exactly


class PredictRequest(BaseModel):
    records: List[FeatureVector]


class PredictResponse(BaseModel):
    probabilities: List[float]
    predictions: List[int]
