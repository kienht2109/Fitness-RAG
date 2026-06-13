from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, StringConstraints

from app.analysis.insight import AnalysisService, get_analysis_service
from app.analysis.models import WorkoutRecord

router = APIRouter(prefix="/analysis", tags=["analysis"])


class AnalysisQueryRequest(BaseModel):
    user_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    history: list[WorkoutRecord]
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]


class AnalysisQueryResponse(BaseModel):
    insight: str
    summary: dict[str, Any]


@router.post("/query", response_model=AnalysisQueryResponse)
async def query_analysis(
    request: AnalysisQueryRequest,
    service: Annotated[AnalysisService, Depends(get_analysis_service)],
) -> AnalysisQueryResponse:
    result = await service.query(
        user_id=request.user_id,
        history=[workout.model_dump(mode="json") for workout in request.history],
        question=request.question,
    )
    return AnalysisQueryResponse(insight=result.insight, summary=result.summary)
