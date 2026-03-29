"""Tests for EmailTriage Pydantic models.

Covers:
  - Valid action construction for all 7 tools
  - model_validator enforcement (required fields per tool)
  - Invalid enum values
  - Observation and State round-trip serialization
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Allow running tests from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (  # noqa: E402
    EmailMetadata,
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
    InboxSummary,
    RewardBreakdown,
    ScoreSummary,
)


# ---------------------------------------------------------------------------
# EmailTriageAction — valid construction
# ---------------------------------------------------------------------------


class TestEmailTriageActionValid:
    def test_read_email(self):
        a = EmailTriageAction(tool="read_email", email_id="e001")
        assert a.tool == "read_email"
        assert a.email_id == "e001"

    def test_classify_email(self):
        a = EmailTriageAction(
            tool="classify_email", email_id="e001", department="Billing"
        )
        assert a.department == "Billing"

    def test_set_priority_critical(self):
        a = EmailTriageAction(
            tool="set_priority", email_id="e006", priority="Critical"
        )
        assert a.priority == "Critical"

    def test_set_priority_all_levels(self):
        for level in ("Critical", "High", "Medium", "Low"):
            a = EmailTriageAction(
                tool="set_priority", email_id="e001", priority=level
            )
            assert a.priority == level

    def test_add_tags_single(self):
        a = EmailTriageAction(
            tool="add_tags", email_id="e001", tags=["complaint"]
        )
        assert "complaint" in a.tags

    def test_add_tags_multiple(self):
        a = EmailTriageAction(
            tool="add_tags",
            email_id="e001",
            tags=["billing-dispute", "complaint", "refund"],
        )
        assert len(a.tags) == 3

    def test_draft_response(self):
        a = EmailTriageAction(
            tool="draft_response",
            email_id="e001",
            response_text="Dear customer, thank you for contacting us. Best regards.",
        )
        assert a.response_text is not None

    def test_mark_spam(self):
        a = EmailTriageAction(tool="mark_spam", email_id="e007")
        assert a.tool == "mark_spam"
        assert a.department is None

    def test_escalate_email(self):
        a = EmailTriageAction(
            tool="escalate_email",
            email_id="e006",
            escalation_reason="Production system down, needs L3 escalation.",
        )
        assert a.escalation_reason is not None


# ---------------------------------------------------------------------------
# EmailTriageAction — validation errors (missing required fields)
# ---------------------------------------------------------------------------


class TestEmailTriageActionValidationErrors:
    def test_read_email_missing_email_id(self):
        with pytest.raises(ValidationError) as exc_info:
            EmailTriageAction(tool="read_email")
        assert "email_id" in str(exc_info.value).lower()

    def test_classify_email_missing_department(self):
        with pytest.raises(ValidationError):
            EmailTriageAction(tool="classify_email", email_id="e001")

    def test_set_priority_missing_priority(self):
        with pytest.raises(ValidationError):
            EmailTriageAction(tool="set_priority", email_id="e001")

    def test_add_tags_missing_tags(self):
        with pytest.raises(ValidationError):
            EmailTriageAction(tool="add_tags", email_id="e001")

    def test_add_tags_empty_list(self):
        with pytest.raises(ValidationError):
            EmailTriageAction(tool="add_tags", email_id="e001", tags=[])

    def test_draft_response_missing_text(self):
        with pytest.raises(ValidationError):
            EmailTriageAction(tool="draft_response", email_id="e001")

    def test_escalate_missing_reason(self):
        with pytest.raises(ValidationError):
            EmailTriageAction(tool="escalate_email", email_id="e001")


# ---------------------------------------------------------------------------
# Invalid enum values
# ---------------------------------------------------------------------------


class TestEmailTriageActionInvalidEnums:
    def test_invalid_tool(self):
        with pytest.raises(ValidationError):
            EmailTriageAction(tool="delete_email", email_id="e001")

    def test_invalid_department(self):
        with pytest.raises(ValidationError):
            EmailTriageAction(
                tool="classify_email",
                email_id="e001",
                department="Finance",
            )

    def test_invalid_priority(self):
        with pytest.raises(ValidationError):
            EmailTriageAction(
                tool="set_priority", email_id="e001", priority="Extreme"
            )

    def test_invalid_tag(self):
        with pytest.raises(ValidationError):
            EmailTriageAction(
                tool="add_tags",
                email_id="e001",
                tags=["not-a-real-tag"],
            )


# ---------------------------------------------------------------------------
# Action round-trip serialization
# ---------------------------------------------------------------------------


class TestActionSerialization:
    def test_model_dump_excludes_none(self):
        a = EmailTriageAction(tool="read_email", email_id="e001")
        dumped = a.model_dump(exclude_none=True)
        assert "department" not in dumped
        assert "priority" not in dumped
        assert dumped["tool"] == "read_email"

    def test_model_validate_roundtrip(self):
        original = EmailTriageAction(
            tool="classify_email",
            email_id="e001",
            department="Billing",
        )
        reloaded = EmailTriageAction.model_validate(original.model_dump())
        assert reloaded == original


# ---------------------------------------------------------------------------
# EmailTriageState defaults
# ---------------------------------------------------------------------------


class TestEmailTriageState:
    def test_default_state(self):
        s = EmailTriageState()
        assert s.task_id == ""
        assert s.done is False
        assert s.cumulative_reward == 0.0
        assert s.unread_email_ids == []

    def test_state_with_values(self):
        s = EmailTriageState(
            task_id="routing_easy",
            unread_email_ids=["e001", "e002"],
            remaining_steps=18,
            cumulative_reward=0.42,
        )
        assert s.task_id == "routing_easy"
        assert len(s.unread_email_ids) == 2
