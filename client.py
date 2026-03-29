"""Typed async client for the EmailTriage benchmark.

Usage (async)::

    from emailtriage.client import EmailTriageEnv
    from emailtriage.models import EmailTriageAction

    async with EmailTriageEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task_id="routing_easy")
        obs = result.observation

        result = await env.step(
            EmailTriageAction(tool="read_email", email_id="e001")
        )
        print(result.observation.last_tool_result)

Usage (sync)::

    with EmailTriageEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(task_id="routing_easy")
"""

from __future__ import annotations

from typing import Dict

try:
    from openenv.core.env_server.client import EnvClient, StepResult
except ImportError:  # pragma: no cover
    # Minimal stubs so the module is importable without openenv-core
    from typing import Generic, TypeVar

    _A = TypeVar("_A")
    _O = TypeVar("_O")
    _S = TypeVar("_S")

    class StepResult(Generic[_O]):  # type: ignore[no-redef]
        def __init__(self, observation: _O, reward: float, done: bool) -> None:
            self.observation = observation
            self.reward = reward
            self.done = done

    class EnvClient(Generic[_A, _O, _S]):  # type: ignore[no-redef]
        def __init__(self, base_url: str = "http://localhost:8000") -> None:
            self.base_url = base_url

try:
    from emailtriage.models import (
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriageState,
    )
except ImportError:
    from models import EmailTriageAction, EmailTriageObservation, EmailTriageState


class EmailTriageEnv(
    EnvClient[EmailTriageAction, EmailTriageObservation, EmailTriageState]
):
    """Persistent session client for the EmailTriage OpenEnv environment.

    Wraps :class:`openenv.core.env_server.client.EnvClient` with concrete
    type parameters and serialization/deserialization logic.
    """

    def _step_payload(self, action: EmailTriageAction) -> Dict:
        """Serialize *action* to the HTTP request body."""
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: Dict
    ) -> StepResult[EmailTriageObservation]:
        """Deserialize the HTTP response into a typed :class:`StepResult`."""
        obs = EmailTriageObservation.model_validate(payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict) -> EmailTriageState:
        """Deserialize the state payload."""
        return EmailTriageState.model_validate(payload)


# Backwards-compatible alias (matches the template-generated name pattern)
EmailTriageClient = EmailTriageEnv
