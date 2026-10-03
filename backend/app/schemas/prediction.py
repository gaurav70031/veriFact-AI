from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class PredictRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=500, description="Article headline")
    content: str = Field(..., min_length=20, description="Article body text")

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "Breaking: Scientists discover new planet",
                "content": "NASA researchers announced today that they have discovered a new planet...",
            }
        }
    }


class PredictResponse(BaseModel):
    id: int
    label: str               # "FAKE" | "REAL"
    is_fake: bool
    confidence: float        # 0.0 – 1.0
    fake_probability: float
    real_probability: float
    model_version: str
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionRecord(BaseModel):
    id: int
    title: Optional[str]
    content: str
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    is_fake: bool
    model_version: str
    created_at: datetime

    model_config = {"from_attributes": True}


class StatsResponse(BaseModel):
    total_predictions: int
    fake_count: int
    real_count: int
    fake_percentage: float
    real_percentage: float
    average_confidence: float
