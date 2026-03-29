"""FastAPI application for the EmailTriage OpenEnv environment.

Exposes:
    Standard OpenEnv endpoints  (reset / step / state / health / web)
        — provided by openenv.core.env_server.http_server.create_app

    Benchmark-specific endpoints:
        GET /tasks    — task descriptors and JSON schemas
        GET /grader   — deterministic grader report for an episode
        GET /baseline — run scripted or LLM baseline and return scores
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install with: pip install 'openenv-core[core]'"
    ) from exc

try:
    from emailtriage.baseline import DEFAULT_BASELINE_MODEL, run_baseline
    from emailtriage.email_core import EmailTaskCatalog, EpisodeRegistry
    from emailtriage.models import (
        BaselineReport,
        EmailTriageAction,
        EmailTriageObservation,
        GraderReport,
        TasksResponse,
    )
    from emailtriage.server.email_triage_environment import EmailTriageEnvironment
except ImportError:
    from baseline import DEFAULT_BASELINE_MODEL, run_baseline
    from email_core import EmailTaskCatalog, EpisodeRegistry
    from models import (
        BaselineReport,
        EmailTriageAction,
        EmailTriageObservation,
        GraderReport,
        TasksResponse,
    )
    from server.email_triage_environment import EmailTriageEnvironment

# ---------------------------------------------------------------------------
# Singletons (shared across all concurrent sessions)
# ---------------------------------------------------------------------------

CATALOG = EmailTaskCatalog()
REGISTRY = EpisodeRegistry()


def _max_concurrent_envs() -> int:
    return int(os.getenv("EMAILTRIAGE_MAX_CONCURRENT_ENVS", "4"))


def _env_factory() -> EmailTriageEnvironment:
    return EmailTriageEnvironment(catalog=CATALOG, registry=REGISTRY)


# ---------------------------------------------------------------------------
# In-process session (used by /baseline to avoid loopback HTTP calls)
# ---------------------------------------------------------------------------


class _InProcessSession:
    """Async context manager that drives the environment directly in-process."""

    def __init__(self) -> None:
        self._env = _env_factory()

    async def __aenter__(self) -> "_InProcessSession":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    async def reset(self, **kwargs: object) -> SimpleNamespace:
        observation = self._env.reset(**kwargs)
        return SimpleNamespace(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    async def step(self, action: EmailTriageAction) -> SimpleNamespace:
        observation = self._env.step(action)
        return SimpleNamespace(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

app: FastAPI = create_app(
    _env_factory,
    EmailTriageAction,
    EmailTriageObservation,
    env_name="email-triage-env",
    max_concurrent_envs=_max_concurrent_envs(),
)

# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Send browsers to the interactive playground."""
    return RedirectResponse(url="/web/", status_code=307)


@app.get("/web", include_in_schema=False)
async def web_root() -> RedirectResponse:
    """Avoid proxy-generated absolute redirects that can downgrade HTTPS."""
    return RedirectResponse(url="/web/", status_code=307)


@app.get("/landing", include_in_schema=False, response_class=HTMLResponse)
async def landing_page() -> HTMLResponse:
    """Human-readable landing page for direct browser visits."""
    return HTMLResponse(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>EmailTriage — OpenEnv</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 700px; margin: 60px auto; padding: 0 20px; }
    h1   { font-size: 1.8rem; }
    a.btn { display: inline-block; margin: 6px 4px; padding: 8px 18px; border-radius: 6px;
            text-decoration: none; font-weight: 600; }
    .primary   { background: #2563eb; color: #fff; }
    .secondary { background: #f1f5f9; color: #1e293b; }
    code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>EmailTriage OpenEnv</h1>
  <p>
    Deterministic benchmark for customer-support email triage — routing,
    priority assignment, spam detection, and response drafting.
  </p>
  <p>
    <a class="btn primary"  href="/web/">Open Playground</a>
    <a class="btn secondary" href="/tasks">View Tasks</a>
    <a class="btn secondary" href="/docs">API Docs</a>
    <a class="btn secondary" href="/baseline?backend=scripted">Run Scripted Baseline</a>
  </p>
  <ul>
    <li><code>GET /tasks</code> — list the 3 benchmark tasks and JSON schemas.</li>
    <li><code>GET /grader?episode_id=…</code> — fetch a deterministic grader report.</li>
    <li><code>GET /baseline?backend=scripted</code> — run without an API key.</li>
    <li><code>GET /baseline?backend=openai&amp;model=gpt-4o-mini</code> — LLM baseline.</li>
  </ul>
</body>
</html>"""
    )


# ---------------------------------------------------------------------------
# Benchmark endpoints
# ---------------------------------------------------------------------------


@app.get("/tasks", response_model=TasksResponse, tags=["Benchmark"])
async def tasks() -> TasksResponse:
    """Return the 3 benchmark tasks with their action and observation schemas."""
    return CATALOG.build_tasks_response()


@app.get("/grader", response_model=GraderReport, tags=["Benchmark"])
async def grader(
    episode_id: str = Query(..., description="Episode identifier returned by reset()")
) -> GraderReport:
    """Return the deterministic grader report for a completed or in-progress episode."""
    report = REGISTRY.get(episode_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No grader report found for episode_id='{episode_id}'. "
            "Ensure you called reset() and passed the correct episode_id.",
        )
    return report


@app.get("/baseline", response_model=BaselineReport, tags=["Benchmark"])
async def baseline(
    request: Request,
    model: str = Query(
        default=os.getenv("OPENAI_MODEL", DEFAULT_BASELINE_MODEL),
        description="Model identifier for the LLM baseline (ignored for scripted backend).",
    ),
    backend: Literal["auto", "openai", "scripted"] = Query(
        default=os.getenv("EMAILTRIAGE_BASELINE_BACKEND", "auto"),
        description=(
            "'scripted' — rule-based, no API key needed. "
            "'openai' — requires OPENAI_API_KEY env var. "
            "'auto' — uses scripted if OPENAI_API_KEY is absent."
        ),
    ),
) -> BaselineReport:
    """Run the baseline against all 3 tasks and return aggregate scores.

    Use ``backend=scripted`` for a fully offline run (reproducible, no LLM).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if backend == "openai" and not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY must be set before calling /baseline with backend=openai.",
        )

    async def _grader_fetcher(episode_id: str) -> GraderReport:
        report = REGISTRY.get(episode_id)
        if report is None:
            raise RuntimeError(
                f"Missing grader report for episode_id={episode_id}"
            )
        return report

    return await run_baseline(
        model=model,
        api_key=api_key,
        backend=backend,
        session_factory=_InProcessSession,
        grader_fetcher=_grader_fetcher,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EmailTriage OpenEnv server.")
    parser.add_argument(
        "--host", default=os.getenv("EMAILTRIAGE_HOST", "0.0.0.0")
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", os.getenv("EMAILTRIAGE_PORT", "8000"))),
    )
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
