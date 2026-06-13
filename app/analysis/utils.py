"""Small statistical and date helpers for workout analysis."""

from datetime import date

from app.analysis.constants import MINIMUM_TRAINING_GAP_DAYS


def date_range(workout_dates: set[str]) -> dict[str, str] | None:
    if not workout_dates:
        return None
    return {"start": min(workout_dates), "end": max(workout_dates)}


def detect_training_gaps(
    workout_dates: set[str], minimum_days: int = MINIMUM_TRAINING_GAP_DAYS
) -> list[dict[str, object]]:
    ordered_dates = sorted(date.fromisoformat(value) for value in workout_dates)
    gaps: list[dict[str, object]] = []
    for previous, following in zip(ordered_dates, ordered_dates[1:]):
        days_without_training = (following - previous).days - 1
        if days_without_training >= minimum_days:
            gaps.append(
                {
                    "after_date": previous.isoformat(),
                    "before_date": following.isoformat(),
                    "days_without_training": days_without_training,
                }
            )
    return gaps


def linear_regression(x_values: list[int], y_values: list[float]) -> tuple[float, float]:
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")

    if len(x_values) < 2:
        raise ValueError("At least two data points are required.")

    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)

    x_variance = sum((value - x_mean) ** 2 for value in x_values)
    if x_variance == 0:
        return 0.0, 0.0

    covariance = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )

    slope = covariance / x_variance

    predicted = [y_mean + slope * (x_value - x_mean) for x_value in x_values]

    residual_sum = sum(
        (actual - estimate) ** 2 for actual, estimate in zip(y_values, predicted, strict=True)
    )

    total_sum = sum((value - y_mean) ** 2 for value in y_values)

    if total_sum == 0:
        r_squared = 1.0
    else:
        r_squared = 1 - residual_sum / total_sum

    return slope, max(0.0, min(1.0, r_squared))
