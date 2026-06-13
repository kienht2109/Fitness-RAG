from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/analysis", tags=["analysis"])


class AnalysisQueryRequest(BaseModel):
    user_id: str = Field(min_length=1)
    history: list[dict[str, Any]]
    question: str = Field(min_length=1)


class AnalysisQueryResponse(BaseModel):
    insight: str
    summary: dict[str, Any]


@router.post("/query", response_model=AnalysisQueryResponse)
async def query_analysis(_: AnalysisQueryRequest) -> AnalysisQueryResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Workout history analysis has not been implemented yet.",
    )
