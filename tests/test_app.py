"""Smoke tests for the EmailTriage FastAPI endpoints.

Tests:
  GET /tasks      — structure and content
  GET /grader     — known episode, unknown episode, missing param
  GET /baseline   — scripted backend (no API key), openai without key → 503
  GET /landing    — HTML response

These tests run against the real FastAPI app via TestClient (in-process,
no network required).  They deliberately avoid mocking the environment so
that any regression in the core engine is surfaced here too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import app and singletons directly
from server.app import CATALOG, REGISTRY, app  # noqa: E402
from email_core import EpisodeRuntime  # noqa: E402
from models import EmailTriageAction  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Reuse a single TestClient across all tests in this module."""
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def episode_id() -> str:
    """Create a real episode and return its ID for /grader tests."""
    runtime = EpisodeRuntime(catalog=CATALOG, task_id="routing_easy")
    # Take one action so the episode has a meaningful state
    runtime.apply_action(EmailTriageAction(tool="read_email", email_id="e001"))
    report = runtime.compute_grader_report()
    REGISTRY.store(report)
    return report.episode_id


# ---------------------------------------------------------------------------
# GET /tasks
# ---------------------------------------------------------------------------


class TestTasksEndpoint:
    def test_returns_200(self, client: TestClient):
        assert client.get("/tasks").status_code == 200

    def test_returns_three_tasks(self, client: TestClient):
        data = client.get("/tasks").json()
        assert len(data["tasks"]) == 3

    def test_task_ids_present(self, client: TestClient):
        ids = [t["task_id"] for t in client.get("/tasks").json()["tasks"]]
        assert "routing_easy" in ids
        assert "priority_medium" in ids
        assert "drafting_hard" in ids

    def test_task_difficulties(self, client: TestClient):
        difficulties = {
            t["task_id"]: t["difficulty"]
            for t in client.get("/tasks").json()["tasks"]
        }
        assert difficulties["routing_easy"] == "easy"
        assert difficulties["priority_medium"] == "medium"
        assert difficulties["drafting_hard"] == "hard"

    def test_includes_action_schema(self, client: TestClient):
        data = client.get("/tasks").json()
        assert "action_schema" in data
        assert data["action_schema"]  # non-empty

    def test_includes_observation_schema(self, client: TestClient):
        data = client.get("/tasks").json()
        assert "observation_schema" in data
        assert data["observation_schema"]

    def test_email_counts(self, client: TestClient):
        tasks = {t["task_id"]: t for t in client.get("/tasks").json()["tasks"]}
        assert tasks["routing_easy"]["email_count"] == 5
        assert tasks["priority_medium"]["email_count"] == 6
        assert tasks["drafting_hard"]["email_count"] == 4

    def test_step_budgets_present(self, client: TestClient):
        for task in client.get("/tasks").json()["tasks"]:
            assert task["step_budget"] > 0


# ---------------------------------------------------------------------------
# GET /grader
# ---------------------------------------------------------------------------


class TestGraderEndpoint:
    def test_known_episode_200(self, client: TestClient, episode_id: str):
        r = client.get(f"/grader?episode_id={episode_id}")
        assert r.status_code == 200

    def test_known_episode_fields(self, client: TestClient, episode_id: str):
        data = client.get(f"/grader?episode_id={episode_id}").json()
        assert data["episode_id"] == episode_id
        assert "score" in data
        assert "task_id" in data
        assert "email_grades" in data

    def test_score_in_range(self, client: TestClient, episode_id: str):
        score = client.get(f"/grader?episode_id={episode_id}").json()["score"]
        assert 0.0 <= score <= 1.0

    def test_unknown_episode_404(self, client: TestClient):
        r = client.get("/grader?episode_id=totally-fake-id")
        assert r.status_code == 404

    def test_missing_episode_id_422(self, client: TestClient):
        r = client.get("/grader")
        assert r.status_code == 422

    def test_email_grades_count(self, client: TestClient, episode_id: str):
        grades = client.get(f"/grader?episode_id={episode_id}").json()["email_grades"]
        assert len(grades) == 5  # routing_easy has 5 emails


# ---------------------------------------------------------------------------
# GET /baseline
# ---------------------------------------------------------------------------


class TestBaselineEndpoint:
    def test_scripted_backend_200(self, client: TestClient):
        r = client.get("/baseline?backend=scripted")
        assert r.status_code == 200

    def test_scripted_returns_three_tasks(self, client: TestClient):
        data = client.get("/baseline?backend=scripted").json()
        assert len(data["tasks"]) == 3

    def test_scripted_overall_score_range(self, client: TestClient):
        data = client.get("/baseline?backend=scripted").json()
        assert 0.0 <= data["overall_score"] <= 1.0

    def test_scripted_task_ids(self, client: TestClient):
        data = client.get("/baseline?backend=scripted").json()
        task_ids = [t["task_id"] for t in data["tasks"]]
        assert "routing_easy" in task_ids
        assert "priority_medium" in task_ids
        assert "drafting_hard" in task_ids

    def test_scripted_model_name(self, client: TestClient):
        data = client.get("/baseline?backend=scripted").json()
        assert data["model"] == "scripted-baseline"

    def test_scripted_task_scores_in_range(self, client: TestClient):
        tasks = client.get("/baseline?backend=scripted").json()["tasks"]
        for t in tasks:
            assert 0.0 <= t["score"] <= 1.0
            assert t["step_count"] > 0

    def test_openai_without_key_503(self, client: TestClient, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        r = client.get("/baseline?backend=openai")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# GET /landing
# ---------------------------------------------------------------------------


class TestLandingPage:
    def test_landing_returns_html(self, client: TestClient):
        r = client.get("/landing")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_landing_contains_env_name(self, client: TestClient):
        body = client.get("/landing").text
        assert "EmailTriage" in body
