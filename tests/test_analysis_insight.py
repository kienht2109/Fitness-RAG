from typing import Any

import anyio
from langchain_core.runnables import RunnableLambda

from app.analysis.insight import (
    NOT_ENOUGH_HISTORY_INSIGHT,
    AnalysisService,
)
from app.analysis.models import AnalysisInsight
from app.analysis.prompting import ANALYSIS_PROMPT
from app.analysis.summary import AnalysisIntent, classify_analysis_intent


def _history() -> list[dict[str, Any]]:
    return [
        {
            "date": "2026-01-01",
            "exercise": "Bench Press",
            "sets": [{"reps": 8, "weight": 70, "unit": "kg"}],
        },
        {
            "date": "2026-02-01",
            "exercise": "Bench Press",
            "sets": [{"reps": 8, "weight": 75, "unit": "kg"}],
        },
    ]


def test_intent_classifier_uses_supported_categories() -> None:
    assert classify_analysis_intent("How is my bench progressing?") is AnalysisIntent.TREND
    assert classify_analysis_intent("What have I neglected?") is AnalysisIntent.NEGLECT
    assert classify_analysis_intent("Is my chest and back balanced?") is AnalysisIntent.BALANCE
    assert (
        classify_analysis_intent("What should I add to my plan?") is AnalysisIntent.PLAN_SUGGESTION
    )


def test_analysis_service_passes_only_structured_summary_to_generation() -> None:
    calls: list[dict[str, Any]] = []

    def generate(inputs: dict[str, Any]) -> AnalysisInsight:
        calls.append(inputs)
        return AnalysisInsight(insight="Bench estimated 1RM increased across the recorded dates.")

    service = AnalysisService(RunnableLambda(generate))
    result = anyio.run(
        lambda: service.query(
            user_id="user_a",
            history=_history(),
            question="How is my Bench Press progressing?",
        )
    )

    assert result.summary["intent"] == "trend"
    assert result.summary["exercise_trends"]["Bench Press"]["strength"]["percent_change"] > 0
    assert "summary_json" in calls[0]
    assert '"weight":70' not in calls[0]["summary_json"]
    assert '"estimated_1rm_kg"' in calls[0]["summary_json"]


def test_analysis_service_skips_generation_for_empty_history() -> None:
    def fail_if_called(_: Any) -> AnalysisInsight:
        raise AssertionError("The insight chain must not run without history")

    service = AnalysisService(RunnableLambda(fail_if_called))
    result = anyio.run(
        lambda: service.query(user_id="user_a", history=[], question="How am I doing?")
    )

    assert result.insight == NOT_ENOUGH_HISTORY_INSIGHT
    assert result.summary == {}


def test_analysis_prompt_contains_summary_not_raw_history_placeholder() -> None:
    prompt = ANALYSIS_PROMPT.invoke(
        {
            "user_id": "user_a",
            "intent": "trend",
            "question": "How am I progressing?",
            "summary_json": '{"training_day_count":2}',
        }
    )

    rendered = str(prompt)
    assert "Deterministic analysis summary" in rendered
    assert "training_day_count" in rendered
    assert "raw workout history is not" in rendered.lower()
