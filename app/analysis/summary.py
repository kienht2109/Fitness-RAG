"""Question-aware summaries built from deterministic workout calculations."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from app.analysis.processing import build_workout_summary


class AnalysisIntent(StrEnum):
    TREND = "trend"
    NEGLECT = "neglect"
    BALANCE = "balance"
    PLAN_SUGGESTION = "plan_suggestion"


def classify_analysis_intent(question: str) -> AnalysisIntent:
    normalized = question.casefold()
    if any(term in normalized for term in ("plan", "program", "what should", "recommend")):
        return AnalysisIntent.PLAN_SUGGESTION
    if any(term in normalized for term in ("balance", "imbalance", "ratio", "overtrain")):
        return AnalysisIntent.BALANCE
    if any(term in normalized for term in ("neglect", "missing", "skip", "rarely", "never")):
        return AnalysisIntent.NEGLECT
    return AnalysisIntent.TREND


def build_analysis_summary(
    history: list[dict[str, Any]], question: str
) -> tuple[AnalysisIntent, dict[str, Any]]:
    """Build a compact summary focused on the question's deterministic intent."""
    intent = classify_analysis_intent(question)
    complete = build_workout_summary(history)
    aggregation = complete["aggregation"]
    trends = complete["trends"]
    exercise_names = list(trends["exercises"])
    mentioned = [name for name in exercise_names if name.casefold() in question.casefold()]
    selected_names = mentioned or exercise_names

    selected_trends = {
        name: _compact_exercise_trend(trends["exercises"][name]) for name in selected_names
    }
    selected_aggregates = {
        name: aggregation["by_exercise"][name]
        for name in selected_names
        if name in aggregation["by_exercise"]
    }

    summary: dict[str, Any] = {
        "intent": intent.value,
        "canonical_weight_unit": aggregation["canonical_weight_unit"],
        "date_range": aggregation["date_range"],
        "training_day_count": aggregation["training_day_count"],
        "mixed_units_normalized": aggregation["mixed_units_normalized"],
        "original_units": aggregation["original_units"],
        "gaps": trends["gaps"],
        "unknown_exercises": sorted(
            name
            for name, stats in aggregation["by_exercise"].items()
            if stats["muscle_group"] == "unknown"
        ),
        "exercise_aggregates": selected_aggregates,
        "exercise_trends": selected_trends,
    }

    if intent in {
        AnalysisIntent.NEGLECT,
        AnalysisIntent.BALANCE,
        AnalysisIntent.PLAN_SUGGESTION,
    }:
        summary["muscle_group_aggregates"] = aggregation["by_muscle_group"]
        summary["rarely_trained_exercises"] = sorted(
            name
            for name, stats in aggregation["by_exercise"].items()
            if stats["session_count"] <= 3
        )
        expected_groups = {"chest", "back", "shoulders", "quadriceps", "posterior_chain"}
        summary["missing_muscle_groups"] = sorted(
            expected_groups - aggregation["by_muscle_group"].keys()
        )

    return intent, summary


def _compact_exercise_trend(trend: dict[str, Any]) -> dict[str, Any]:
    return {
        "muscle_group": trend["muscle_group"],
        "session_count": trend["session_count"],
        "weighted_session_count": trend["weighted_session_count"],
        "bodyweight_session_count": trend["bodyweight_session_count"],
        "mixed_units_normalized": trend["mixed_units_normalized"],
        "original_units": trend["original_units"],
        "likely_deload_dates": trend["likely_deload_dates"],
        "strength": trend["strength"],
        "bodyweight_reps": trend["bodyweight_reps"],
        "progression_pattern": trend["progression_pattern"],
    }
