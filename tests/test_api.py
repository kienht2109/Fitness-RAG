from fastapi.testclient import TestClient

from app.analysis.insight import AnalysisResult, get_analysis_service
from app.api.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_feature_routes_are_registered() -> None:
    paths = client.get("/openapi.json").json()["paths"]

    assert "/rag/query" in paths
    assert "/analysis/query" in paths
    assert "/agent/query" in paths


def test_analysis_request_validates_workout_history_shape() -> None:
    response = client.post(
        "/analysis/query",
        json={
            "user_id": "user_a",
            "question": "How is my bench progressing?",
            "history": [
                {
                    "date": "2026-01-02",
                    "exercise": "Bench Press",
                    "sets": [{"reps": 8, "weight": 70, "unit": "stone"}],
                }
            ],
        },
    )

    assert response.status_code == 422


class FakeAnalysisService:
    async def query(self, *, user_id, history, question) -> AnalysisResult:
        assert user_id == "user_a"
        assert question == "How is my bench progressing?"
        assert history[0]["exercise"] == "Bench Press"
        return AnalysisResult(
            insight="Your estimated bench strength is progressing.",
            summary={"intent": "trend", "training_day_count": 1},
        )


def test_analysis_endpoint_returns_generated_insight_and_summary() -> None:
    app.dependency_overrides[get_analysis_service] = lambda: FakeAnalysisService()
    client = TestClient(app)
    try:
        response = client.post(
            "/analysis/query",
            json={
                "user_id": "user_a",
                "question": "How is my bench progressing?",
                "history": [
                    {
                        "date": "2026-01-02",
                        "exercise": "Bench Press",
                        "sets": [{"reps": 8, "weight": 70, "unit": "kg"}],
                    }
                ],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "insight": "Your estimated bench strength is progressing.",
        "summary": {"intent": "trend", "training_day_count": 1},
    }
