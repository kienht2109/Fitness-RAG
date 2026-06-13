import pytest

from app.analysis.processing import (
    aggregate_workout_data,
    detect_exercise_trends,
    normalize_weight_to_kg,
    normalize_workout_history,
)


@pytest.fixture(scope="module")
def histories() -> dict[str, list[dict]]:
    return {
        "user_a": [
            _workout("2026-01-03", "Pull-Up", [(10, 0), (9, 0), (8, 0)]),
            _workout("2026-01-07", "Pull-Up", [(10, 0), (10, 0), (9, 0)]),
            _workout("2026-01-14", "Pull-Up", [(10, 5), (8, 5), (7, 5)]),
            _workout("2026-01-20", "Bench Press", [(8, 75), (8, 75), (7, 75), (6, 75)]),
            _workout("2026-01-21", "Pull-Up", [(10, 5), (9, 5), (8, 5)]),
            _workout("2026-01-27", "Bench Press", [(5, 60), (5, 60)]),
            _workout("2026-02-03", "Bench Press", [(8, 75), (8, 75), (8, 75), (7, 75)]),
            _workout("2026-02-04", "Pull-Up", [(10, 7.5), (9, 7.5), (8, 7.5)]),
            _workout("2026-03-04", "Pull-Up", [(10, 10), (9, 10), (8, 10)]),
            _workout("2026-03-10", "Bench Press", [(8, 82.5), (7, 82.5), (6, 82.5)]),
        ],
        "user_b": [
            _workout("2026-01-05", "Bench Press", [(10, 110), (8, 110)], unit="lb"),
            _workout("2026-02-09", "Bench Press", [(8, 125), (7, 125)], unit="lb"),
            _workout("2026-02-23", "Bench Press", [(10, 55), (9, 55)]),
            _workout("2026-03-16", "Bench Press", [(10, 60), (9, 60)]),
        ],
    }


def _workout(
    workout_date: str,
    exercise: str,
    sets: list[tuple[int, float]],
    *,
    unit: str = "kg",
) -> dict:
    return {
        "date": workout_date,
        "exercise": exercise,
        "sets": [{"reps": reps, "weight": weight, "unit": unit} for reps, weight in sets],
    }


def test_normalizes_lb_and_preserves_zero_weight() -> None:
    history = [
        {
            "date": "2026-01-01",
            "exercise": "Pull-Up",
            "sets": [
                {"reps": 5, "weight": 100, "unit": "lb"},
                {"reps": 8, "weight": 0, "unit": "kg"},
            ],
        }
    ]

    normalized = normalize_workout_history(history)

    assert normalized[0]["sets"][0]["weight_kg"] == pytest.approx(45.359237)
    assert normalized[0]["sets"][1]["weight_kg"] == 0
    assert normalize_weight_to_kg(10, "kg") == 10


def test_aggregation_tracks_weight_volume_and_bodyweight_reps(histories) -> None:
    result = aggregate_workout_data(histories["user_a"])

    pull_ups = result["by_exercise"]["Pull-Up"]
    assert pull_ups["session_count"] == 6
    assert pull_ups["bodyweight_set_count"] == 6
    assert pull_ups["bodyweight_rep_count"] == 56
    assert pull_ups["weight_volume_kg"] > 0
    assert result["by_muscle_group"]["chest"]["weight_volume_kg"] > 0


def test_mixed_units_do_not_create_false_bench_regression(histories) -> None:
    result = detect_exercise_trends(histories["user_b"])
    bench = result["exercises"]["Bench Press"]

    assert bench["mixed_units_normalized"] is True
    assert bench["strength"]["status"] == "progressing"
    assert bench["strength"]["percent_change"] > 0
    assert bench["series"][0]["top_weight_kg"] == pytest.approx(49.9, abs=0.1)
    assert bench["series"][-1]["top_weight_kg"] == 60


def test_deload_is_annotated_and_excluded_from_strength_fit(histories) -> None:
    result = detect_exercise_trends(histories["user_a"])
    bench = result["exercises"]["Bench Press"]

    assert bench["likely_deload_dates"] == ["2026-01-27"]
    assert bench["strength"]["data_points"] == bench["weighted_session_count"] - 1
    assert bench["strength"]["status"] == "progressing"
    assert next(point for point in bench["series"] if point["date"] == "2026-01-27")[
        "is_likely_deload"
    ]


def test_bodyweight_only_exercise_gets_rep_trend() -> None:
    history = [
        {
            "date": "2026-01-01",
            "exercise": "Pull-Up",
            "sets": [{"reps": 5, "weight": 0, "unit": "kg"}],
        },
        {
            "date": "2026-01-08",
            "exercise": "Pull-Up",
            "sets": [{"reps": 8, "weight": 0, "unit": "kg"}],
        },
    ]

    trend = detect_exercise_trends(history)["exercises"]["Pull-Up"]

    assert trend["strength"]["status"] == "not_applicable"
    assert trend["bodyweight_reps"]["status"] == "progressing"
    assert trend["bodyweight_reps"]["percent_change"] == 60


def test_sparse_history_reports_gap_without_treating_it_as_a_session() -> None:
    history = [
        {
            "date": "2026-01-01",
            "exercise": "Squat",
            "sets": [{"reps": 5, "weight": 80, "unit": "kg"}],
        },
        {
            "date": "2026-01-22",
            "exercise": "Squat",
            "sets": [{"reps": 5, "weight": 85, "unit": "kg"}],
        },
    ]

    result = detect_exercise_trends(history)

    assert result["gaps"] == [
        {
            "after_date": "2026-01-01",
            "before_date": "2026-01-22",
            "days_without_training": 20,
        }
    ]
    assert result["exercises"]["Squat"]["strength"]["slope_per_week"] > 0


def test_empty_history_returns_empty_serializable_summaries() -> None:
    assert aggregate_workout_data([])["date_range"] is None
    assert detect_exercise_trends([]) == {
        "canonical_weight_unit": "kg",
        "date_range": None,
        "gaps": [],
        "exercises": {},
    }
