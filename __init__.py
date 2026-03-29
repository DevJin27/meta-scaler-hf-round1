"""EmailTriage — OpenEnv environment for customer support email triage.

Quick start::

    from emailtriage import EmailTriageEnv, EmailTriageAction

    async with EmailTriageEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task_id="routing_easy")
        result = await env.step(
            EmailTriageAction(tool="read_email", email_id="e001")
        )
"""

from __future__ import annotations

try:
    from emailtriage.client import EmailTriageClient, EmailTriageEnv
    from emailtriage.models import (
        BaselineReport,
        BaselineTaskResult,
        EmailGrade,
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriageState,
        GraderReport,
        InboxSummary,
        RewardBreakdown,
        ScoreSummary,
        TaskDescriptor,
        TasksResponse,
    )
except ImportError:
    from client import EmailTriageClient, EmailTriageEnv
    from models import (
        BaselineReport,
        BaselineTaskResult,
        EmailGrade,
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriageState,
        GraderReport,
        InboxSummary,
        RewardBreakdown,
        ScoreSummary,
        TaskDescriptor,
        TasksResponse,
    )

__all__ = [
    # Client
    "EmailTriageEnv",
    "EmailTriageClient",
    # Core types
    "EmailTriageAction",
    "EmailTriageObservation",
    "EmailTriageState",
    # Grader / benchmark
    "GraderReport",
    "EmailGrade",
    "BaselineReport",
    "BaselineTaskResult",
    "TaskDescriptor",
    "TasksResponse",
    # Observation sub-models
    "InboxSummary",
    "RewardBreakdown",
    "ScoreSummary",
]
