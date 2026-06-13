from fastapi.testclient import TestClient

from app.analysis.history import UserNotFoundError, WorkoutHistoryUnavailableError
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


def test_analysis_request_rejects_client_supplied_history() -> None:
    response = client.post(
        "/analysis/query",
        json={
            "user_id": "user_a",
            "question": "How is my bench progressing?",
            "history": [],
        },
    )

    assert response.status_code == 422


class FakeAnalysisService:
    async def query(self, *, user_id, question) -> AnalysisResult:
        assert user_id == "user_a"
        assert question == "How is my bench progressing?"
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
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "insight": "Your estimated bench strength is progressing.",
        "summary": {"intent": "trend", "training_day_count": 1},
    }


class FailingAnalysisService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def query(self, *, user_id, question) -> AnalysisResult:
        raise self.error


def test_analysis_endpoint_returns_404_for_unknown_user() -> None:
    service = FailingAnalysisService(UserNotFoundError("Unknown user_id: user_c"))
    app.dependency_overrides[get_analysis_service] = lambda: service
    client = TestClient(app)
    try:
        response = client.post(
            "/analysis/query",
            json={"user_id": "user_c", "question": "How am I progressing?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 404


def test_analysis_endpoint_returns_503_for_unavailable_history() -> None:
    service = FailingAnalysisService(WorkoutHistoryUnavailableError("invalid source"))
    app.dependency_overrides[get_analysis_service] = lambda: service
    client = TestClient(app)
    try:
        response = client.post(
            "/analysis/query",
            json={"user_id": "user_a", "question": "How am I progressing?"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
