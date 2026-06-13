from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.agent.orchestrator import AgentService, get_agent_service

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]


class AgentQueryResponse(BaseModel):
    answer: str
    tools_used: list[str]


@router.post("/query", response_model=AgentQueryResponse)
async def query_agent(
    request: AgentQueryRequest,
    service: Annotated[AgentService, Depends(get_agent_service)],
) -> AgentQueryResponse:
    result = await service.query(
        user_id=request.user_id,
        question=request.question,
    )
    return AgentQueryResponse(answer=result.answer, tools_used=result.tools_used)
