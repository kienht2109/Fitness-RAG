from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.routes_agent import router as agent_router
from app.api.routes_analysis import router as analysis_router
from app.api.routes_health import router as health_router
from app.api.routes_rag import router as rag_router
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Resolve and validate configuration once during application startup.
    get_settings()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Fitness RAG, workout analysis, and coach-assist API.",
        lifespan=lifespan,
    )
    application.include_router(health_router)
    application.include_router(rag_router)
    application.include_router(analysis_router)
    application.include_router(agent_router)
    return application


app = create_app()
