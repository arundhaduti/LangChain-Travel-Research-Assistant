from pydantic import BaseModel, Field, field_validator
from typing import List

class DailyPlan(BaseModel):
    day: int
    activities: List[str]
    estimated_cost: float

class TravelPlan(BaseModel):
    destination: str
    total_days: int = Field(..., ge=1, le=10)
    total_budget: float
    plans: List[DailyPlan]

    @field_validator("total_budget")
    @classmethod
    def check_budget(cls, v):
        if v <= 0:
            raise ValueError("Budget must be positive")
        return v

    @field_validator("plans")
    @classmethod
    def validate_days(cls, v, info):
        total_days = info.data.get("total_days")
        if total_days and len(v) != total_days:
            raise ValueError("Number of daily plans must match total_days")
        return v
