"""Public models for the EmailTriage OpenEnv benchmark."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:  # pragma: no cover — falls back for offline unit tests
    from pydantic import BaseModel as _Base

    class Action(_Base):  # type: ignore[no-redef]
        pass

    class Observation(_Base):  # type: ignore[no-redef]
        pass

    class State(_Base):  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Domain literals
# ---------------------------------------------------------------------------

ToolName = Literal[
    "read_email",
    "classify_email",
    "set_priority",
    "add_tags",
    "draft_response",
    "mark_spam",
    "escalate_email",
]

DepartmentLabel = Literal[
    "Billing",
    "Technical Support",
    "Account Management",
    "Sales",
    "HR",
    "General",
]

PriorityLabel = Literal["Critical", "High", "Medium", "Low"]

TagLabel = Literal[
    "complaint",
    "inquiry",
    "refund",
    "password-reset",
    "order-status",
    "technical-issue",
    "account-closure",
    "payment-failed",
    "spam",
    "feedback",
    "billing-dispute",
    "service-outage",
    "feature-request",
]

EmailStatus = Literal["unread", "classified", "escalated", "spam", "responded"]

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class EmailMetadata(BaseModel):
    """Visible metadata attached to each email."""

    account_tier: Literal["standard", "premium", "enterprise"] = Field(
        ..., description="Customer account tier"
    )
    is_vip: bool = Field(..., description="Whether the customer is a VIP")
    revenue_impact: float = Field(
        ..., description="Estimated revenue impact in USD (positive = at risk, negative = churn)"
    )


class EmailSummary(BaseModel):
    """Inbox-level summary of an email (no body)."""

    email_id: str = Field(..., description="Email identifier")
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Sender email address")
    status: EmailStatus = Field(..., description="Current processing status")
    priority: Optional[PriorityLabel] = Field(default=None, description="Assigned priority")
    department: Optional[DepartmentLabel] = Field(default=None, description="Assigned department")


class EmailView(BaseModel):
    """Full visible content of an email (revealed after read_email)."""

    email_id: str = Field(..., description="Email identifier")
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Full email body")
    sender: str = Field(..., description="Sender email address")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    metadata: EmailMetadata = Field(..., description="Account metadata")
    status: EmailStatus = Field(..., description="Current processing status")
    tags: List[str] = Field(default_factory=list, description="Applied tags")


class InboxSummary(BaseModel):
    """Aggregate inbox state exposed to the agent."""

    total_emails: int = Field(..., description="Total emails in the task")
    unread: int = Field(..., description="Unread emails")
    classified: int = Field(..., description="Classified but not yet responded")
    escalated: int = Field(..., description="Escalated emails")
    spam: int = Field(..., description="Emails marked as spam")
    responded: int = Field(..., description="Emails with a drafted response")
    emails: List[EmailSummary] = Field(
        default_factory=list, description="Per-email inbox listing"
    )


class RewardBreakdown(BaseModel):
    """Detailed reward payload carried in every observation."""

    step_reward: float = Field(..., description="Scalar reward for the current step")
    bonuses: Dict[str, float] = Field(
        default_factory=dict, description="Named positive reward contributions"
    )
    penalties: Dict[str, float] = Field(
        default_factory=dict, description="Named negative reward contributions"
    )
    cumulative_reward: float = Field(
        ..., description="Total reward accumulated in this episode"
    )
    provisional_episode_score: float = Field(
        ..., description="Current grader score estimate (0.0–1.0)"
    )


class ScoreSummary(BaseModel):
    """Compact running score snapshot."""

    provisional_episode_score: float = Field(
        ..., description="Current grader score estimate"
    )
    emails_processed: int = Field(..., description="Number of emails no longer unread")
    emails_remaining: int = Field(..., description="Number of unread emails")
    cumulative_reward: float = Field(..., description="Total reward accumulated so far")


# ---------------------------------------------------------------------------
# Task / grader models
# ---------------------------------------------------------------------------


class TaskDescriptor(BaseModel):
    """Public description of one benchmark task."""

    task_id: str = Field(..., description="Stable task identifier")
    title: str = Field(..., description="Human-readable title")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Difficulty bucket"
    )
    objective: str = Field(..., description="What the agent must accomplish")
    email_count: int = Field(..., description="Number of emails in the task")
    step_budget: int = Field(..., description="Maximum steps before the episode ends")
    allowed_tools: List[ToolName] = Field(
        default_factory=list, description="Tools available in this task"
    )


class EmailGrade(BaseModel):
    """Per-email grading result."""

    email_id: str = Field(..., description="Email identifier")
    status: EmailStatus = Field(..., description="Final processing status")
    score: float = Field(..., description="Email score (0.0–1.0)")
    component_scores: Dict[str, float] = Field(
        default_factory=dict, description="Named sub-scores (routing, priority, tag_f1, etc.)"
    )
    notes: List[str] = Field(
        default_factory=list, description="Deterministic grader feedback notes"
    )


class GraderReport(BaseModel):
    """Episode-level grading report returned by /grader."""

    episode_id: str = Field(..., description="Episode identifier")
    task_id: str = Field(..., description="Task identifier")
    title: str = Field(..., description="Task title")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Task difficulty"
    )
    completed: bool = Field(..., description="Whether the episode ended (done=True)")
    score: float = Field(..., description="Episode score (0.0–1.0)")
    cumulative_reward: float = Field(..., description="Total scalar reward accumulated")
    step_count: int = Field(..., description="Steps taken")
    email_grades: List[EmailGrade] = Field(
        default_factory=list, description="Per-email grading results"
    )


class BaselineTaskResult(BaseModel):
    """Per-task result from the baseline runner."""

    task_id: str = Field(..., description="Task identifier")
    episode_id: str = Field(..., description="Episode identifier used in the run")
    score: float = Field(..., description="Final task score")
    cumulative_reward: float = Field(..., description="Final cumulative reward")
    step_count: int = Field(..., description="Steps used")


class BaselineReport(BaseModel):
    """Top-level result returned by the baseline runner."""

    model: str = Field(..., description="Model or backend used for the baseline")
    overall_score: float = Field(..., description="Mean score across all tasks")
    tasks: List[BaselineTaskResult] = Field(
        default_factory=list, description="Per-task baseline results"
    )


class TasksResponse(BaseModel):
    """Response payload for the /tasks endpoint."""

    tasks: List[TaskDescriptor] = Field(
        default_factory=list, description="Benchmark tasks"
    )
    action_schema: Dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for EmailTriageAction"
    )
    observation_schema: Dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for EmailTriageObservation"
    )


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class EmailTriageAction(Action):
    """Structured tool call issued by the agent."""

    tool: ToolName = Field(..., description="Tool to invoke")
    email_id: Optional[str] = Field(default=None, description="Target email identifier")
    department: Optional[DepartmentLabel] = Field(
        default=None, description="Department for classify_email"
    )
    priority: Optional[PriorityLabel] = Field(
        default=None, description="Priority for set_priority"
    )
    tags: Optional[List[TagLabel]] = Field(
        default=None, description="Tags to apply with add_tags"
    )
    response_text: Optional[str] = Field(
        default=None, description="Customer-facing response draft"
    )
    escalation_reason: Optional[str] = Field(
        default=None, description="Reason supplied when escalating"
    )

    @model_validator(mode="after")
    def validate_tool_payload(self) -> "EmailTriageAction":
        required_fields: Dict[str, List[str]] = {
            "read_email": ["email_id"],
            "classify_email": ["email_id", "department"],
            "set_priority": ["email_id", "priority"],
            "add_tags": ["email_id", "tags"],
            "draft_response": ["email_id", "response_text"],
            "mark_spam": ["email_id"],
            "escalate_email": ["email_id", "escalation_reason"],
        }
        missing = [
            f
            for f in required_fields[self.tool]
            if getattr(self, f) in (None, "", [])
        ]
        if missing:
            raise ValueError(
                f"Tool '{self.tool}' requires the following fields: {', '.join(missing)}"
            )
        return self


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class EmailTriageObservation(Observation):
    """Full observation returned after each action."""

    episode_id: str = Field(..., description="Episode identifier")
    task_id: str = Field(..., description="Task identifier")
    task_title: str = Field(..., description="Task title")
    task_difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Task difficulty"
    )
    task_objective: str = Field(..., description="What the agent must accomplish")
    allowed_tools: List[ToolName] = Field(
        default_factory=list, description="Tools available in this episode"
    )
    inbox_summary: InboxSummary = Field(
        ..., description="Aggregate inbox state"
    )
    active_email: Optional[EmailView] = Field(
        default=None, description="Full content of the last read email"
    )
    last_tool_result: Dict[str, Any] = Field(
        default_factory=dict, description="Structured result of the most recent tool call"
    )
    remaining_step_budget: int = Field(
        ..., description="Steps remaining in this episode"
    )
    score_summary: ScoreSummary = Field(
        ..., description="Compact running score snapshot"
    )
    reward_breakdown: RewardBreakdown = Field(
        ..., description="Detailed reward payload for this step"
    )
    # Shortcut fields accessed by _InProcessSession and EnvClient
    reward: float = Field(default=0.0, description="Step reward (mirrors reward_breakdown.step_reward)")
    done: bool = Field(default=False, description="Whether the episode has ended")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class EmailTriageState(State):
    """Safe state summary for session-aware clients."""

    task_id: str = Field(default="", description="Current task identifier")
    task_title: str = Field(default="", description="Current task title")
    active_email_id: Optional[str] = Field(
        default=None, description="Last email passed to read_email"
    )
    unread_email_ids: List[str] = Field(
        default_factory=list, description="Email IDs still unread"
    )
    processed_email_ids: List[str] = Field(
        default_factory=list, description="Email IDs no longer unread"
    )
    remaining_steps: int = Field(default=0, description="Steps remaining in the episode")
    cumulative_reward: float = Field(
        default=0.0, description="Total reward accumulated so far"
    )
    done: bool = Field(default=False, description="Whether the episode has ended")
