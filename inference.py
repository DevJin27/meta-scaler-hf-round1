"""inference.py — EmailTriage OpenEnv agent for the Meta Scalability Challenge.

Follows the sample inference.py spec exactly:
  - API_BASE_URL, MODEL_NAME, HF_TOKEN loaded from environment.
  - Defaults only for API_BASE_URL and MODEL_NAME (not HF_TOKEN).
  - All LLM calls use the OpenAI client configured via these variables.
  - Stdout logs follow the required START/STEP/END structured format.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment configuration — matches the pre-submission checklist exactly
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "<your-api-base-url>")
MODEL_NAME   = os.getenv("MODEL_NAME", "<your-active-model-name>")
HF_TOKEN     = os.getenv("HF_TOKEN")

# Optional — only required when using from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# ---------------------------------------------------------------------------
# OpenAI client — all LLM calls go through this single client instance
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ---------------------------------------------------------------------------
# Structured logging helpers (START / STEP / END)
# ---------------------------------------------------------------------------


def _log(event: str, **kwargs: Any) -> None:
    """Emit a structured log line to stdout.

    Format:   [EVENT] key=value key=value …
    Events:   START, STEP, END
    """
    parts = " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[{event}] {parts}", flush=True)


def log_start(task_id: str, episode_id: Optional[str] = None) -> None:
    _log("START", task_id=task_id, episode_id=episode_id or "pending")


def log_step(
    step: int,
    tool: str,
    email_id: Optional[str],
    reward: float,
    done: bool,
) -> None:
    _log(
        "STEP",
        step=step,
        tool=tool,
        email_id=email_id or "N/A",
        reward=f"{reward:.4f}",
        done=done,
    )


def log_end(episode_id: str, score: float, steps: int, cumulative_reward: float) -> None:
    _log(
        "END",
        episode_id=episode_id,
        score=f"{score:.4f}",
        steps=steps,
        cumulative_reward=f"{cumulative_reward:.4f}",
    )


# ---------------------------------------------------------------------------
# System prompt + message builders
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert customer-support triage agent working inside the EmailTriage
benchmark environment.

Your objective: {task_objective}

Available tools: {allowed_tools}

For every unread email in the inbox follow this workflow:
1. Call read_email to reveal the full email content.
2. If the email is obviously spam, call mark_spam and move to the next email.
3. Otherwise:
   a. classify_email  — route it to the correct department.
   b. set_priority    — assign Critical / High / Medium / Low.
   c. add_tags        — apply up to 3 relevant semantic tags (skip if no tags fit).
   d. draft_response  — write a professional customer-facing reply (hard task only).
4. Repeat until all emails are processed or you run out of steps.

You MUST respond with a **single JSON object** that matches this schema exactly:
{action_schema}

Rules:
- Output ONLY the JSON object — no markdown fences, no explanation.
- Never repeat an action you have already taken for the same email.
- Prioritise Critical and High-priority emails first.
"""

# JSON schema for EmailTriageAction (kept minimal; full schema injected at runtime)
_ACTION_SCHEMA_STUB = json.dumps(
    {
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "enum": [
                    "read_email",
                    "classify_email",
                    "set_priority",
                    "add_tags",
                    "draft_response",
                    "mark_spam",
                    "escalate_email",
                ],
            },
            "email_id": {"type": "string"},
            "department": {
                "type": "string",
                "enum": [
                    "Billing",
                    "Technical Support",
                    "Account Management",
                    "Sales",
                    "HR",
                    "General",
                ],
            },
            "priority": {
                "type": "string",
                "enum": ["Critical", "High", "Medium", "Low"],
            },
            "tags": {"type": "array", "items": {"type": "string"}},
            "response_text": {"type": "string"},
            "escalation_reason": {"type": "string"},
        },
        "required": ["tool"],
    },
    indent=2,
)


def _build_system_prompt(task_objective: str, allowed_tools: List[str]) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        task_objective=task_objective,
        allowed_tools=", ".join(allowed_tools),
        action_schema=_ACTION_SCHEMA_STUB,
    )


def _inbox_user_message(obs_dict: Dict[str, Any]) -> str:
    inbox = obs_dict.get("inbox_summary", {})
    remaining = obs_dict.get("remaining_step_budget", "?")
    return (
        f"Current inbox state:\n{json.dumps(inbox, default=str, indent=2)}\n\n"
        f"Remaining step budget: {remaining}\n\n"
        "Next action?"
    )


def _tool_result_message(obs_dict: Dict[str, Any]) -> str:
    tool_result = obs_dict.get("last_tool_result", {})
    inbox = obs_dict.get("inbox_summary", {})
    remaining = obs_dict.get("remaining_step_budget", "?")
    reward = obs_dict.get("reward", 0.0)
    return (
        f"Tool result: {json.dumps(tool_result, default=str)}\n"
        f"Step reward: {reward:.4f}\n"
        f"Remaining steps: {remaining}\n"
        f"Inbox: {json.dumps(inbox, default=str)}\n\n"
        "Next action?"
    )


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------


def _call_llm(messages: List[Dict[str, str]]) -> str:
    """Send messages to the LLM and return the raw text content."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


def _parse_action(content: str) -> Optional[Dict[str, Any]]:
    """Strip optional markdown fences and parse the JSON action."""
    clean = content.strip()
    if clean.startswith("```"):
        # remove opening fence line
        lines = clean.split("\n")
        clean = "\n".join(lines[1:])
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]
    try:
        return json.loads(clean.strip())
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Environment HTTP session helpers
# ---------------------------------------------------------------------------


def _env_reset(env_base_url: str, task_id: str) -> Dict[str, Any]:
    """POST /reset to the environment server and return the observation dict."""
    import urllib.request

    payload = json.dumps({"task_id": task_id}).encode()
    req = urllib.request.Request(
        f"{env_base_url}/reset",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _env_step(env_base_url: str, episode_id: str, action: Dict[str, Any]) -> Dict[str, Any]:
    """POST /step to the environment server and return the observation dict."""
    import urllib.request

    payload = json.dumps({"episode_id": episode_id, "action": action}).encode()
    req = urllib.request.Request(
        f"{env_base_url}/step",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

# Default environment URL — override with ENV_BASE_URL environment variable.
_DEFAULT_ENV_URL = "http://localhost:8000"


def run_agent(
    task_id: str = "routing_easy",
    env_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single episode of the EmailTriage benchmark using the LLM agent.

    Parameters
    ----------
    task_id:
        One of: ``routing_easy``, ``priority_medium``, ``drafting_hard``.
    env_base_url:
        Base URL of the running EmailTriage OpenEnv server.  Falls back to
        the ``ENV_BASE_URL`` environment variable, then ``localhost:8000``.

    Returns
    -------
    dict
        Summary with ``episode_id``, ``final_score``, ``steps``, and
        ``cumulative_reward``.
    """
    env_url = env_base_url or os.getenv("ENV_BASE_URL", _DEFAULT_ENV_URL)

    # ---- START ----
    log_start(task_id)

    # Reset the environment
    obs_dict = _env_reset(env_url, task_id)
    episode_id: str = obs_dict["episode_id"]
    task_objective: str = obs_dict.get("task_objective", "")
    allowed_tools: List[str] = obs_dict.get("allowed_tools", [])
    done: bool = obs_dict.get("done", False)

    log_start(task_id, episode_id)  # re-emit with real episode_id

    # Build conversation history
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _build_system_prompt(task_objective, allowed_tools)},
        {"role": "user", "content": _inbox_user_message(obs_dict)},
    ]

    step_num = 0
    max_steps = obs_dict.get("remaining_step_budget", 40)

    while not done and step_num < max_steps:
        # ---- LLM call ----
        content = _call_llm(messages)
        messages.append({"role": "assistant", "content": content})

        action = _parse_action(content)

        if action is None:
            # Ask the LLM to self-correct
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your response was not valid JSON. "
                        "Respond with a single valid EmailTriageAction JSON object."
                    ),
                }
            )
            continue

        # ---- Environment step ----
        try:
            obs_dict = _env_step(env_url, episode_id, action)
        except Exception as exc:
            messages.append(
                {
                    "role": "user",
                    "content": f"Environment error: {exc}. Try a different action.",
                }
            )
            continue

        step_num += 1
        done = obs_dict.get("done", False)
        reward: float = obs_dict.get("reward", 0.0)

        # ---- STEP log ----
        log_step(
            step=step_num,
            tool=action.get("tool", "unknown"),
            email_id=action.get("email_id"),
            reward=reward,
            done=done,
        )

        # Continue conversation
        messages.append({"role": "user", "content": _tool_result_message(obs_dict)})

    # ---- END ----
    score_summary = obs_dict.get("score_summary", {})
    final_score: float = score_summary.get("provisional_episode_score", 0.0)
    cumulative_reward: float = score_summary.get("cumulative_reward", 0.0)

    log_end(
        episode_id=episode_id,
        score=final_score,
        steps=step_num,
        cumulative_reward=cumulative_reward,
    )

    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "final_score": final_score,
        "steps": step_num,
        "cumulative_reward": cumulative_reward,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the EmailTriage LLM agent against one or all tasks."
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=["routing_easy", "priority_medium", "drafting_hard", "all"],
        help="Task to run (default: all).",
    )
    parser.add_argument(
        "--env-url",
        default=os.getenv("ENV_BASE_URL", _DEFAULT_ENV_URL),
        help="EmailTriage server base URL (default: http://localhost:8000).",
    )
    args = parser.parse_args()

    tasks = (
        ["routing_easy", "priority_medium", "drafting_hard"]
        if args.task == "all"
        else [args.task]
    )

    results = []
    for task_id in tasks:
        result = run_agent(task_id=task_id, env_base_url=args.env_url)
        results.append(result)
        print(json.dumps(result, indent=2), file=sys.stderr)

    if results:
        overall = sum(r["final_score"] for r in results) / len(results)
        print(
            json.dumps({"overall_score": overall, "task_results": results}, indent=2),
            file=sys.stderr,
        )
