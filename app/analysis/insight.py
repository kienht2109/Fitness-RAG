"""Workout analysis service and LLM-backed insight generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from anyio import to_thread
from langchain_core.runnables import Runnable

from app.analysis.history import JsonWorkoutHistoryRepository, WorkoutHistoryRepository
from app.analysis.models import AnalysisInsight
from app.analysis.prompting import build_insight_chain
from app.analysis.summary import build_analysis_summary
from app.core.ai import create_chat_model
from app.core.config import Settings, get_settings

NOT_ENOUGH_HISTORY_INSIGHT = "There is not enough workout history to analyze yet."


@dataclass(frozen=True)
class AnalysisResult:
    insight: str
    summary: dict[str, Any]


class AnalysisService:
    def __init__(
        self,
        history_repository: WorkoutHistoryRepository,
        insight_chain: Runnable[Any, AnalysisInsight],
    ) -> None:
        self.history_repository = history_repository
        self.insight_chain = insight_chain

    async def query(
        self,
        *,
        user_id: str,
        question: str,
    ) -> AnalysisResult:
        normalized_user_id = user_id.strip()
        normalized_question = question.strip()
        if not normalized_user_id:
            raise ValueError("user_id must not be empty")
        if not normalized_question:
            raise ValueError("Question must not be empty")
        user = await to_thread.run_sync(
            lambda: self.history_repository.get_user(normalized_user_id)
        )
        history = [workout.model_dump(mode="json") for workout in user.workouts]
        if not history:
            return AnalysisResult(insight=NOT_ENOUGH_HISTORY_INSIGHT, summary={})

        intent, summary = build_analysis_summary(history, normalized_question)
        summary["user"] = {
            "user_id": normalized_user_id,
            "name": user.name,
            "profile": user.profile,
        }
        generated = await self.insight_chain.ainvoke(
            {
                "user_id": normalized_user_id,
                "intent": intent.value,
                "question": normalized_question,
                "summary_json": json.dumps(summary, sort_keys=True, separators=(",", ":")),
            }
        )
        insight = (
            generated
            if isinstance(generated, AnalysisInsight)
            else AnalysisInsight.model_validate(generated)
        )
        return AnalysisResult(insight=insight.insight.strip(), summary=summary)


def create_analysis_service(settings: Settings | None = None) -> AnalysisService:
    settings = settings or get_settings()
    chat_model = create_chat_model(settings)
    return AnalysisService(
        JsonWorkoutHistoryRepository(settings.workout_history_path),
        build_insight_chain(chat_model),
    )


@lru_cache
def get_analysis_service() -> AnalysisService:
    return create_analysis_service()
