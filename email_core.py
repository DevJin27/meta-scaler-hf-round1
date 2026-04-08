"""Shared deterministic engine for the EmailTriage benchmark.

This module is the single source of truth for:
  - Email dataset loading and task definitions
  - Per-episode state tracking
  - Reward computation (fully deterministic, no LLM)
  - Grader report generation
  - Episode registry (thread-safe storage)
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from .models import (
        BaselineReport,
        BaselineTaskResult,
        EmailGrade,
        EmailMetadata,
        EmailSummary,
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriageState,
        EmailView,
        GraderReport,
        InboxSummary,
        RewardBreakdown,
        ScoreSummary,
        TaskDescriptor,
        TasksResponse,
    )
except ImportError:  # pragma: no cover
    from models import (
        BaselineReport,
        BaselineTaskResult,
        EmailGrade,
        EmailMetadata,
        EmailSummary,
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriageState,
        EmailView,
        GraderReport,
        InboxSummary,
        RewardBreakdown,
        ScoreSummary,
        TaskDescriptor,
        TasksResponse,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_ORDER: Tuple[str, ...] = ("routing_easy", "priority_medium", "drafting_hard")

GLOBAL_INVALID_ACTION_PENALTY: float = -0.02
GLOBAL_DUPLICATE_ACTION_PENALTY: float = -0.02

PRIORITY_TIERS: Dict[str, int] = {
    "Critical": 0,
    "High": 1,
    "Medium": 2,
    "Low": 3,
}

# ---------------------------------------------------------------------------
# Text utilities (all deterministic)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """Lowercase and replace punctuation with spaces."""
    return re.sub(r"[^\w\s]", " ", text.lower())


def tokenize(text: str) -> List[str]:
    """Split normalized text into word tokens."""
    return normalize_text(text).split()


def keyword_coverage(text: str, keywords: List[str]) -> float:
    """Fraction of *keywords* present (as substrings) in *text* after normalization.

    Uses substring match so 'apologize' matches 'apologizes'.
    """
    if not keywords:
        return 0.0
    norm = normalize_text(text)
    matched = sum(1 for kw in keywords if normalize_text(kw) in norm)
    return matched / len(keywords)


def priority_distance(pred: str, true: str) -> int:
    """Absolute tier distance between two priority labels."""
    return abs(PRIORITY_TIERS.get(pred, 2) - PRIORITY_TIERS.get(true, 2))


def tag_f1(pred_tags: List[str], true_tags: List[str]) -> float:
    """Micro-averaged F1 between predicted and ground-truth tag sets."""
    if not true_tags:
        return 1.0 if not pred_tags else 0.0
    if not pred_tags:
        return 0.0
    pred_set = set(pred_tags)
    true_set = set(true_tags)
    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class GroundTruth:
    department: str
    priority: str
    tags: List[str]
    is_spam: bool
    kb_keywords: List[str]
    response_must_include: List[str]
    should_escalate: bool = False


@dataclass
class EmailRecord:
    email_id: str
    subject: str
    body: str
    sender: str
    timestamp: str
    metadata: Dict[str, Any]
    ground_truth: GroundTruth


@dataclass
class TaskDefinition:
    task_id: str
    title: str
    difficulty: str
    objective: str
    email_ids: List[str]
    step_budget: int
    allowed_tools: List[str]


@dataclass
class EmailProgress:
    """Mutable per-email state for one episode."""

    email_id: str
    status: str = "unread"
    classified_department: Optional[str] = None
    classified_priority: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    response_draft: Optional[str] = None
    is_marked_spam: bool = False
    is_escalated: bool = False
    escalation_reason: Optional[str] = None
    has_been_read: bool = False
    # Each entry is "tool:email_id" — used for duplicate detection
    actions_taken: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class EmailTaskCatalog:
    """Loads emails and task definitions from data/emails.json.

    Thread-safe after construction (read-only).
    """

    def __init__(self, data_path: Optional[Path] = None) -> None:
        if data_path is None:
            data_path = Path(__file__).parent / "data" / "emails.json"
        with open(data_path, encoding="utf-8") as fh:
            raw = json.load(fh)

        self._emails: Dict[str, EmailRecord] = {}
        for item in raw["emails"]:
            gt_raw = item["_ground_truth"]
            gt = GroundTruth(
                department=gt_raw["department"],
                priority=gt_raw["priority"],
                tags=gt_raw["tags"],
                is_spam=gt_raw["is_spam"],
                kb_keywords=gt_raw["kb_keywords"],
                response_must_include=gt_raw["response_must_include"],
                should_escalate=gt_raw.get("should_escalate", False),
            )
            self._emails[item["email_id"]] = EmailRecord(
                email_id=item["email_id"],
                subject=item["subject"],
                body=item["body"],
                sender=item["sender"],
                timestamp=item["timestamp"],
                metadata=item["metadata"],
                ground_truth=gt,
            )

        self._tasks: Dict[str, TaskDefinition] = {}
        for task_raw in raw["tasks"]:
            self._tasks[task_raw["task_id"]] = TaskDefinition(
                task_id=task_raw["task_id"],
                title=task_raw["title"],
                difficulty=task_raw["difficulty"],
                objective=task_raw["objective"],
                email_ids=task_raw["email_ids"],
                step_budget=task_raw["step_budget"],
                allowed_tools=task_raw["allowed_tools"],
            )

    def get_email(self, email_id: str) -> Optional[EmailRecord]:
        return self._emails.get(email_id)

    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[TaskDefinition]:
        return [self._tasks[tid] for tid in TASK_ORDER if tid in self._tasks]

    def build_tasks_response(self) -> TasksResponse:
        descriptors = [
            TaskDescriptor(
                task_id=t.task_id,
                title=t.title,
                difficulty=t.difficulty,
                objective=t.objective,
                email_count=len(t.email_ids),
                step_budget=t.step_budget,
                allowed_tools=t.allowed_tools,
            )
            for t in self.list_tasks()
        ]
        return TasksResponse(
            tasks=descriptors,
            action_schema=EmailTriageAction.model_json_schema(),
            observation_schema=EmailTriageObservation.model_json_schema(),
        )


# ---------------------------------------------------------------------------
# Episode runtime
# ---------------------------------------------------------------------------


class EpisodeRuntime:
    """Manages all mutable state for a single agent episode.

    One EpisodeRuntime is created per reset() call.
    """

    def __init__(self, catalog: EmailTaskCatalog, task_id: str) -> None:
        self.episode_id: str = str(uuid4())
        self.catalog = catalog
        task = catalog.get_task(task_id)
        if task is None:
            raise ValueError(f"Unknown task_id: {task_id!r}. Available: {list(TASK_ORDER)}")
        self.task: TaskDefinition = task

        self.progress: Dict[str, EmailProgress] = {
            eid: EmailProgress(email_id=eid) for eid in task.email_ids
        }
        self.step_count: int = 0
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self._active_email_id: Optional[str] = None
        self._last_tool_result: Dict[str, Any] = {
            "message": "Episode started. Use read_email to open an email."
        }

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------

    def apply_action(
        self, action: EmailTriageAction
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """Process *action* and return (step_reward, bonuses, penalties).

        Mutates episode state in place.
        """
        bonuses: Dict[str, float] = {}
        penalties: Dict[str, float] = {}
        step_reward: float = 0.0

        email_id = action.email_id

        # Validate the email belongs to this task
        if email_id not in self.progress:
            penalties["invalid_email_id"] = GLOBAL_INVALID_ACTION_PENALTY
            step_reward += GLOBAL_INVALID_ACTION_PENALTY
            self._last_tool_result = {
                "error": f"Email '{email_id}' is not in this task's inbox.",
                "available": list(self.progress.keys()),
            }
            self.cumulative_reward += step_reward
            return step_reward, bonuses, penalties

        ep = self.progress[email_id]
        record = self.catalog.get_email(email_id)
        assert record is not None  # always present after catalog init
        gt = record.ground_truth

        # Duplicate action guard
        action_key = f"{action.tool}:{email_id}"
        if action_key in ep.actions_taken:
            penalties["duplicate_action"] = GLOBAL_DUPLICATE_ACTION_PENALTY
            step_reward += GLOBAL_DUPLICATE_ACTION_PENALTY
            self._last_tool_result = {
                "warning": f"Action '{action.tool}' already performed on {email_id}. No further reward."
            }
            self.cumulative_reward += step_reward
            return step_reward, bonuses, penalties

        ep.actions_taken.append(action_key)

        # ---- Dispatch -------------------------------------------------------

        if action.tool == "read_email":
            ep.has_been_read = True
            self._active_email_id = email_id
            self._last_tool_result = {
                "email_id": email_id,
                "subject": record.subject,
                "body": record.body,
                "sender": record.sender,
                "timestamp": record.timestamp,
                "metadata": record.metadata,
            }
            bonuses["email_read"] = 0.01
            step_reward += 0.01

        elif action.tool == "classify_email":
            if not ep.has_been_read:
                penalties["classify_without_read"] = -0.05
                step_reward -= 0.05
            ep.classified_department = action.department
            if action.department == gt.department:
                bonuses["correct_department"] = 0.20
                step_reward += 0.20
                if ep.status == "unread":
                    ep.status = "classified"
            else:
                penalties["wrong_department"] = -0.05
                step_reward -= 0.05
            self._last_tool_result = {
                "email_id": email_id,
                "department_assigned": action.department,
                "correct": action.department == gt.department,
            }

        elif action.tool == "set_priority":
            if not ep.has_been_read:
                penalties["priority_without_read"] = -0.03
                step_reward -= 0.03
            ep.classified_priority = action.priority
            dist = priority_distance(action.priority, gt.priority)
            if dist == 0:
                bonuses["correct_priority"] = 0.15
                step_reward += 0.15
            elif dist == 1:
                bonuses["adjacent_priority"] = 0.05
                step_reward += 0.05
            else:
                penalties["wrong_priority"] = round(-0.05 * dist, 4)
                step_reward -= 0.05 * dist
            # SLA bonus: resolving Critical/High early in the episode
            if gt.priority in ("Critical", "High") and self.step_count <= 15:
                bonuses["sla_timing"] = 0.05
                step_reward += 0.05
            self._last_tool_result = {
                "email_id": email_id,
                "priority_assigned": action.priority,
                "ground_truth_priority": gt.priority,
                "tier_distance": dist,
            }

        elif action.tool == "add_tags":
            if not ep.has_been_read:
                penalties["tags_without_read"] = -0.02
                step_reward -= 0.02
            new_tags: List[str] = list(action.tags or [])
            # Only count tags not already applied
            novel_tags = [t for t in new_tags if t not in ep.tags]
            ep.tags = list(set(ep.tags + new_tags))
            for tag in novel_tags:
                if tag in gt.tags:
                    bonuses[f"correct_tag_{tag}"] = 0.05
                    step_reward += 0.05
                else:
                    penalties[f"wrong_tag_{tag}"] = -0.02
                    step_reward -= 0.02
            f1 = tag_f1(ep.tags, gt.tags)
            self._last_tool_result = {
                "email_id": email_id,
                "tags_applied": ep.tags,
                "tag_f1": round(f1, 3),
            }

        elif action.tool == "draft_response":
            if gt.is_spam:
                penalties["response_to_spam"] = -0.10
                step_reward -= 0.10
                self._last_tool_result = {
                    "warning": "This email is spam — no response should be drafted."
                }
            else:
                if not ep.has_been_read:
                    penalties["response_without_read"] = -0.05
                    step_reward -= 0.05
                ep.response_draft = action.response_text
                ep.status = "responded"
                # Keyword coverage against required phrases
                coverage = keyword_coverage(
                    action.response_text, gt.response_must_include
                )
                response_reward = round(0.40 * coverage, 4)
                bonuses["response_quality"] = response_reward
                step_reward += response_reward
                # Professional tone bonus/penalty
                text_lower = action.response_text.lower()
                has_greeting = any(
                    w in text_lower for w in ("dear", "hello", "hi ", "greetings")
                )
                has_closing = any(
                    w in text_lower
                    for w in ("regards", "sincerely", "thank you", "best wishes")
                )
                if has_greeting and has_closing:
                    bonuses["professional_tone"] = 0.05
                    step_reward += 0.05
                elif not (has_greeting or has_closing):
                    penalties["unprofessional_tone"] = -0.05
                    step_reward -= 0.05
                self._last_tool_result = {
                    "email_id": email_id,
                    "coverage_score": round(coverage, 3),
                    "keywords_found": [
                        kw
                        for kw in gt.response_must_include
                        if normalize_text(kw) in normalize_text(action.response_text)
                    ],
                    "response_drafted": True,
                }

        elif action.tool == "mark_spam":
            ep.is_marked_spam = True
            ep.status = "spam"
            if gt.is_spam:
                bonuses["correct_spam_detection"] = 0.10
                step_reward += 0.10
            else:
                penalties["false_spam_positive"] = -0.10
                step_reward -= 0.10
            self._last_tool_result = {
                "email_id": email_id,
                "marked_spam": True,
                "correct": gt.is_spam,
            }

        elif action.tool == "escalate_email":
            ep.is_escalated = True
            ep.escalation_reason = action.escalation_reason
            ep.status = "escalated"
            if gt.should_escalate:
                bonuses["correct_escalation"] = 0.05
                step_reward += 0.05
            else:
                penalties["unnecessary_escalation"] = -0.03
                step_reward -= 0.03
            self._last_tool_result = {
                "email_id": email_id,
                "escalated": True,
                "reason": action.escalation_reason,
            }

        self.cumulative_reward = round(self.cumulative_reward + step_reward, 6)
        return step_reward, bonuses, penalties

    # ------------------------------------------------------------------
    # Done check
    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        """Episode ends when budget is exhausted or all emails are processed."""
        if self.step_count >= self.task.step_budget:
            return True
        return all(ep.status != "unread" for ep in self.progress.values())

    # ------------------------------------------------------------------
    # Score computation (task-specific, fully deterministic)
    # ------------------------------------------------------------------

    def _compute_task_score(self) -> float:
        """Compute the 0.0–1.0 episode score according to task rules."""
        task_id = self.task.task_id

        if task_id == "routing_easy":
            # Pure routing accuracy
            correct = sum(
                1
                for ep in self.progress.values()
                if ep.classified_department
                == self.catalog.get_email(ep.email_id).ground_truth.department
            )
            return correct / len(self.progress) if self.progress else 0.0

        elif task_id == "priority_medium":
            priority_scores: List[float] = []
            spam_scores: List[float] = []
            tag_scores: List[float] = []

            for ep in self.progress.values():
                gt = self.catalog.get_email(ep.email_id).ground_truth
                if gt.is_spam:
                    spam_scores.append(1.0 if ep.is_marked_spam else 0.0)
                else:
                    if ep.classified_priority:
                        dist = priority_distance(ep.classified_priority, gt.priority)
                        priority_scores.append(max(0.0, 1.0 - dist * 0.34))
                    else:
                        priority_scores.append(0.0)
                    tag_scores.append(tag_f1(ep.tags, gt.tags))

            p_score = (
                sum(priority_scores) / len(priority_scores) if priority_scores else 0.0
            )
            s_score = sum(spam_scores) / len(spam_scores) if spam_scores else 0.0
            t_score = sum(tag_scores) / len(tag_scores) if tag_scores else 0.0
            return 0.5 * p_score + 0.3 * s_score + 0.2 * t_score

        elif task_id == "drafting_hard":
            routing_s: List[float] = []
            priority_s: List[float] = []
            tag_s: List[float] = []
            response_s: List[float] = []

            for ep in self.progress.values():
                gt = self.catalog.get_email(ep.email_id).ground_truth
                routing_s.append(
                    1.0 if ep.classified_department == gt.department else 0.0
                )
                if ep.classified_priority:
                    dist = priority_distance(ep.classified_priority, gt.priority)
                    priority_s.append(max(0.0, 1.0 - dist * 0.34))
                else:
                    priority_s.append(0.0)
                tag_s.append(tag_f1(ep.tags, gt.tags))
                if ep.response_draft:
                    response_s.append(
                        keyword_coverage(ep.response_draft, gt.response_must_include)
                    )
                else:
                    response_s.append(0.0)

            r = sum(routing_s) / len(routing_s) if routing_s else 0.0
            p = sum(priority_s) / len(priority_s) if priority_s else 0.0
            t = sum(tag_s) / len(tag_s) if tag_s else 0.0
            q = sum(response_s) / len(response_s) if response_s else 0.0
            return 0.25 * r + 0.25 * p + 0.10 * t + 0.40 * q

        return 0.0

    # ------------------------------------------------------------------
    # Observation & state builders
    # ------------------------------------------------------------------

    def to_observation(
        self,
        step_reward: float,
        bonuses: Dict[str, float],
        penalties: Dict[str, float],
    ) -> EmailTriageObservation:
        score = self._compute_task_score()

        # Build inbox
        inbox_emails: List[EmailSummary] = []
        counts: Dict[str, int] = {
            "unread": 0,
            "classified": 0,
            "escalated": 0,
            "spam": 0,
            "responded": 0,
        }
        for eid in self.task.email_ids:
            ep = self.progress[eid]
            rec = self.catalog.get_email(eid)
            assert rec is not None
            inbox_emails.append(
                EmailSummary(
                    email_id=eid,
                    subject=rec.subject,
                    sender=rec.sender,
                    status=ep.status,
                    priority=ep.classified_priority,
                    department=ep.classified_department,
                )
            )
            counts[ep.status] = counts.get(ep.status, 0) + 1

        # Build active email view
        active_view: Optional[EmailView] = None
        if self._active_email_id and self._active_email_id in self.progress:
            eid = self._active_email_id
            ep = self.progress[eid]
            rec = self.catalog.get_email(eid)
            assert rec is not None
            active_view = EmailView(
                email_id=eid,
                subject=rec.subject,
                body=rec.body,
                sender=rec.sender,
                timestamp=rec.timestamp,
                metadata=EmailMetadata(**rec.metadata),
                status=ep.status,
                tags=ep.tags,
            )

        processed = (
            counts["classified"]
            + counts["escalated"]
            + counts["spam"]
            + counts["responded"]
        )

        return EmailTriageObservation(
            episode_id=self.episode_id,
            task_id=self.task.task_id,
            task_title=self.task.title,
            task_difficulty=self.task.difficulty,
            task_objective=self.task.objective,
            allowed_tools=self.task.allowed_tools,
            inbox_summary=InboxSummary(
                total_emails=len(self.task.email_ids),
                unread=counts["unread"],
                classified=counts["classified"],
                escalated=counts["escalated"],
                spam=counts["spam"],
                responded=counts["responded"],
                emails=inbox_emails,
            ),
            active_email=active_view,
            last_tool_result=self._last_tool_result,
            remaining_step_budget=self.task.step_budget - self.step_count,
            score_summary=ScoreSummary(
                provisional_episode_score=round(score, 4),
                emails_processed=processed,
                emails_remaining=counts["unread"],
                cumulative_reward=self.cumulative_reward,
            ),
            reward_breakdown=RewardBreakdown(
                step_reward=step_reward,
                bonuses=bonuses,
                penalties=penalties,
                cumulative_reward=self.cumulative_reward,
                provisional_episode_score=round(score, 4),
            ),
            reward=step_reward,
            done=self.done,
        )

    def to_state(self) -> EmailTriageState:
        unread = [eid for eid, ep in self.progress.items() if ep.status == "unread"]
        processed = [eid for eid, ep in self.progress.items() if ep.status != "unread"]
        return EmailTriageState(
            task_id=self.task.task_id,
            task_title=self.task.title,
            active_email_id=self._active_email_id,
            unread_email_ids=unread,
            processed_email_ids=processed,
            remaining_steps=self.task.step_budget - self.step_count,
            cumulative_reward=self.cumulative_reward,
            done=self.done,
        )

    # ------------------------------------------------------------------
    # Grader report
    # ------------------------------------------------------------------

    def compute_grader_report(self) -> GraderReport:
        """Build a deterministic per-email grader report for this episode."""
        email_grades: List[EmailGrade] = []

        for ep in self.progress.values():
            rec = self.catalog.get_email(ep.email_id)
            assert rec is not None
            gt = rec.ground_truth
            notes: List[str] = []
            component_scores: Dict[str, float] = {}

            # Routing
            routing_score = 1.0 if ep.classified_department == gt.department else 0.0
            component_scores["routing"] = routing_score
            if ep.classified_department is None:
                notes.append("No department assigned.")
            elif routing_score == 0.0:
                notes.append(
                    f"Routing: got '{ep.classified_department}', expected '{gt.department}'."
                )

            # Priority
            if ep.classified_priority:
                dist = priority_distance(ep.classified_priority, gt.priority)
                priority_score = max(0.0, 1.0 - dist * 0.34)
            else:
                priority_score = 0.0
                notes.append("No priority assigned.")
            component_scores["priority"] = round(priority_score, 3)

            # Tag F1
            t_f1 = tag_f1(ep.tags, gt.tags)
            component_scores["tag_f1"] = round(t_f1, 3)
            if not ep.tags and gt.tags:
                notes.append(f"No tags applied (expected: {gt.tags}).")

            # Spam detection
            if gt.is_spam:
                spam_score = 1.0 if ep.is_marked_spam else 0.0
                component_scores["spam_detection"] = spam_score
                if not ep.is_marked_spam:
                    notes.append("Spam email not detected.")

            # Response quality
            if ep.response_draft:
                coverage = keyword_coverage(ep.response_draft, gt.response_must_include)
                component_scores["response_quality"] = round(coverage, 3)
            elif gt.response_must_include and self.task.task_id == "drafting_hard":
                notes.append("No response drafted.")
                component_scores["response_quality"] = 0.0

            # Per-task email score
            task_id = self.task.task_id
            if task_id == "routing_easy":
                email_score = routing_score
            elif task_id == "priority_medium":
                if gt.is_spam:
                    email_score = component_scores.get("spam_detection", 0.0)
                else:
                    email_score = (
                        0.5 * priority_score
                        + 0.2 * t_f1
                        + 0.3 * routing_score
                    )
            else:  # drafting_hard
                email_score = (
                    0.25 * routing_score
                    + 0.25 * priority_score
                    + 0.10 * t_f1
                    + 0.40 * component_scores.get("response_quality", 0.0)
                )

            email_grades.append(
                EmailGrade(
                    email_id=ep.email_id,
                    status=ep.status,
                    score=round(email_score, 4),
                    component_scores=component_scores,
                    notes=notes,
                )
            )

        overall_score = self._compute_task_score()

        return GraderReport(
            episode_id=self.episode_id,
            task_id=self.task.task_id,
            title=self.task.title,
            difficulty=self.task.difficulty,
            completed=self.done,
            score=round(overall_score, 4),
            cumulative_reward=round(self.cumulative_reward, 4),
            step_count=self.step_count,
            email_grades=email_grades,
        )


# ---------------------------------------------------------------------------
# Episode registry
# ---------------------------------------------------------------------------


class EpisodeRegistry:
    """Thread-safe storage for episode runtime and grader report data."""

    def __init__(self) -> None:
        self._reports: Dict[str, GraderReport] = {}
        self._runtimes: Dict[str, EpisodeRuntime] = {}
        self._lock = threading.Lock()

    def store(self, report: GraderReport) -> None:
        with self._lock:
            self._reports[report.episode_id] = report

    def store_runtime(self, runtime: EpisodeRuntime) -> None:
        with self._lock:
            self._runtimes[runtime.episode_id] = runtime

    def get(self, episode_id: str) -> Optional[GraderReport]:
        with self._lock:
            return self._reports.get(episode_id)

    def get_runtime(self, episode_id: str) -> Optional[EpisodeRuntime]:
        with self._lock:
            return self._runtimes.get(episode_id)
