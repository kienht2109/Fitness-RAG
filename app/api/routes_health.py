from typing import Literal

from anyio import to_thread
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.core.chroma import create_chroma_client
from app.core.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]


def check_chroma() -> None:
    settings = get_settings()
    create_chroma_client(settings).heartbeat()


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse, summary="Dependency readiness check")
async def ready() -> HealthResponse:
    try:
        await to_thread.run_sync(check_chroma)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chroma is not ready",
        ) from exc

    return HealthResponse(status="ok")
