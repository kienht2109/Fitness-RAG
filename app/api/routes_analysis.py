from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.analysis.history import UserNotFoundError, WorkoutHistoryUnavailableError
from app.analysis.insight import AnalysisService, get_analysis_service

router = APIRouter(prefix="/analysis", tags=["analysis"])


class AnalysisQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
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
    try:
        result = await service.query(user_id=request.user_id, question=request.question)
    except UserNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except WorkoutHistoryUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Workout history is temporarily unavailable.",
        ) from exc
    return AnalysisQueryResponse(insight=result.insight, summary=result.summary)
