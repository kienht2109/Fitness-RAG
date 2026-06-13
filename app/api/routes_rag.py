from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/rag", tags=["rag"])


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1)


class RagSource(BaseModel):
    source_file: str
    section_title: str
    chunk_id: str | None = None


class RagQueryResponse(BaseModel):
    answer: str
    sources: list[RagSource]


@router.post("/query", response_model=RagQueryResponse)
async def query_rag(_: RagQueryRequest) -> RagQueryResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="The RAG pipeline has not been implemented yet.",
    )
