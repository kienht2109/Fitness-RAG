"""Trend metric calculation and progression classification."""

from __future__ import annotations

from datetime import date
from typing import Any
import math


from app.analysis.constants import (
    STRUCTURED_PROGRESSION_R_SQUARED,
    TREND_PERCENT_CHANGE_THRESHOLD,
    TREND_WEEKLY_SLOPE_THRESHOLD,
)
from app.analysis.utils import linear_regression


def calculate_trend_metrics(points: list[dict[str, Any]], *, metric: str) -> dict[str, Any]:
    if not points:
        return {
            "status": "not_applicable",
            "data_points": 0,
            "metric": metric,
        }

    # Ensure chronological order
    points = sorted(points, key=lambda point: point["date"])

    try:
        parsed_dates = [date.fromisoformat(point["date"]) for point in points]
        y_values = [float(point[metric]) for point in points]
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "status": "invalid_data",
            "data_points": len(points),
            "metric": metric,
            "error": str(exc),
        }

    if any(not math.isfinite(value) for value in y_values):
        return {
            "status": "invalid_data",
            "data_points": len(points),
            "metric": metric,
            "error": "Metric values must be finite numbers.",
        }

    if len(points) == 1:
        value = y_values[0]
        return {
            "status": "insufficient_data",
            "data_points": 1,
            "metric": metric,
            "start_date": points[0]["date"],
            "end_date": points[0]["date"],
            "start_value": round(value, 2),
            "end_value": round(value, 2),
        }

    start_date = parsed_dates[0]
    x_values = [(current_date - start_date).days for current_date in parsed_dates]

    slope_per_day, r_squared = linear_regression(x_values, y_values)

    first = y_values[0]
    last = y_values[-1]
    absolute_change = last - first
    percent_change = ((absolute_change / first) * 100) if first != 0 else None

    return {
        "status": _classify_trend(percent_change, slope_per_day, first),
        "data_points": len(points),
        "metric": metric,
        "start_date": points[0]["date"],
        "end_date": points[-1]["date"],
        "start_value": round(first, 2),
        "end_value": round(last, 2),
        "absolute_change": round(absolute_change, 2),
        "percent_change": round(percent_change, 2) if percent_change is not None else None,
        "slope_per_week": round(slope_per_day * 7, 3),
        "r_squared": round(r_squared, 3),
    }


def progression_pattern(strength: dict[str, Any], reps: dict[str, Any]) -> str:
    trend = strength if strength.get("data_points", 0) >= 2 else reps
    if trend.get("data_points", 0) < 2:
        return "insufficient_data"
    if _matches_pattern(trend, "progressing"):
        return "structured_progression"
    if _matches_pattern(trend, "plateau"):
        return "stable"
    if _matches_pattern(trend, "regressing"):
        return "consistent_regression"
    return "inconsistent"


def _classify_trend(percent_change: float | None, slope_per_day: float, start_value: float) -> str:
    if percent_change is None or start_value == 0:
        return "insufficient_data"
    relative_weekly_slope = slope_per_day * 7 / start_value * 100
    if (
        percent_change >= TREND_PERCENT_CHANGE_THRESHOLD
        and relative_weekly_slope > TREND_WEEKLY_SLOPE_THRESHOLD
    ):
        return "progressing"
    if (
        percent_change <= -TREND_PERCENT_CHANGE_THRESHOLD
        and relative_weekly_slope < -TREND_WEEKLY_SLOPE_THRESHOLD
    ):
        return "regressing"
    return "plateau"


def _matches_pattern(trend: dict[str, Any], status: str) -> bool:
    return (
        trend["status"] == status and trend.get("r_squared", 0) >= STRUCTURED_PROGRESSION_R_SQUARED
    )
