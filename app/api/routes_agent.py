from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentQueryRequest(BaseModel):
    user_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class AgentQueryResponse(BaseModel):
    answer: str
    tools_used: list[str]


@router.post("/query", response_model=AgentQueryResponse)
async def query_agent(_: AgentQueryRequest) -> AgentQueryResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="The coach-assist agent has not been implemented yet.",
    )
