from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, StringConstraints

from app.rag.retrieve import RetrievalService, get_retrieval_service

router = APIRouter(prefix="/rag", tags=["rag"])


class RagQueryRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class RagSource(BaseModel):
    source_file: str
    section_title: str
    chunk_id: str | None = None


class RagQueryResponse(BaseModel):
    answer: str
    sources: list[RagSource]


@router.post("/query", response_model=RagQueryResponse)
async def query_rag(
    request: RagQueryRequest,
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> RagQueryResponse:
    result = await service.query(request.question)
    return RagQueryResponse(
        answer=result.answer,
        sources=[RagSource(**source.__dict__) for source in result.sources],
    )
