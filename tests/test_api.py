from fastapi.testclient import TestClient

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
