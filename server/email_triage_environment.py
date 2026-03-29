"""EmailTriage OpenEnv Environment — server-side implementation.

Implements the standard OpenEnv 3-method interface:
    reset(**kwargs) -> EmailTriageObservation
    step(action)    -> EmailTriageObservation
    state()         -> EmailTriageState
"""

from __future__ import annotations

from typing import Optional

try:
    from openenv.core.env_server.environment import Environment
except ImportError:  # pragma: no cover
    # Fallback base class so the module can be imported without openenv-core
    # (used during unit tests).
    class Environment:  # type: ignore[no-redef]
        pass

try:
    from emailtriage.email_core import (
        EmailTaskCatalog,
        EpisodeRegistry,
        EpisodeRuntime,
    )
    from emailtriage.models import (
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriageState,
    )
except ImportError:
    from email_core import EmailTaskCatalog, EpisodeRegistry, EpisodeRuntime
    from models import EmailTriageAction, EmailTriageObservation, EmailTriageState

DEFAULT_TASK = "routing_easy"


class EmailTriageEnvironment(Environment):
    """Stateful OpenEnv environment for email triage.

    One instance is created per concurrent session by *env_factory* in app.py.
    """

    # Each instance owns its own _runtime, so concurrent sessions are safe.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        catalog: EmailTaskCatalog,
        registry: EpisodeRegistry,
    ) -> None:
        self._catalog = catalog
        self._registry = registry
        self._runtime: Optional[EpisodeRuntime] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: str = DEFAULT_TASK,
        seed: Optional[int] = None,
        **kwargs: object,
    ) -> EmailTriageObservation:
        """Start a new episode.

        Parameters
        ----------
        task_id:
            One of ``routing_easy``, ``priority_medium``, or ``drafting_hard``.
            Passed automatically by the TRL trainer from the dataset ``task_id``
            column, or manually when calling via the HTTP client.
        seed:
            Ignored (environment is fully deterministic) but accepted for
            interface compatibility.
        """
        self._runtime = EpisodeRuntime(catalog=self._catalog, task_id=task_id)
        obs = self._runtime.to_observation(step_reward=0.0, bonuses={}, penalties={})
        # Persist initial grader report so /grader is queryable immediately
        self._registry.store(self._runtime.compute_grader_report())
        return obs

    def step(self, action: EmailTriageAction) -> EmailTriageObservation:
        """Execute *action* and return the next observation.

        Raises
        ------
        RuntimeError
            If ``reset()`` has not been called first.
        """
        if self._runtime is None:
            raise RuntimeError(
                "reset() must be called before step(). "
                "Call POST /reset with a task_id to start an episode."
            )

        self._runtime.step_count += 1
        step_reward, bonuses, penalties = self._runtime.apply_action(action)

        # Check terminal condition after the action is applied
        self._runtime.done = self._runtime.is_done()

        obs = self._runtime.to_observation(step_reward, bonuses, penalties)

        # Keep grader report up-to-date after every step
        self._registry.store(self._runtime.compute_grader_report())

        return obs

    def state(self) -> EmailTriageState:
        """Return the current episode state without advancing it."""
        if self._runtime is None:
            return EmailTriageState()
        return self._runtime.to_state()

    def close(self) -> None:
        """Release resources held by this environment instance."""
        self._runtime = None
