from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any

class PredictionRequest(BaseModel):
    age: int = Field(ge=0, le=120, description="年龄，必须在0-120之间")
    visit_frequency: float = Field(ge=0, description="每月访问次数，必须非负")

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    feature_importance: Dict[str, float]

class ErrorResponse(BaseModel):
    error: str
    details: Dict[str, Any] = None
