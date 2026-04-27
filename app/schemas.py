from pydantic import BaseModel, Field


class FoodPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class AnalyzeFoodResponse(BaseModel):
    food_name: str
    confidence: float = Field(ge=0, le=1)
    top_predictions: list[FoodPrediction]
    visual_warnings: list[str]
