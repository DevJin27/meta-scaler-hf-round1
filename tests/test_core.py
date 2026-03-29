"""Tests for the EmailTriage deterministic core engine.

Covers:
  - Text utility functions (normalize, tokenize, coverage, distance, f1)
  - EmailTaskCatalog (data loading, task definitions)
  - EpisodeRuntime (action processing, reward logic, grader reports)
  - EpisodeRegistry (thread-safe storage)
  - Per-task scoring formulae
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from email_core import (  # noqa: E402
    GLOBAL_DUPLICATE_ACTION_PENALTY,
    EmailTaskCatalog,
    EpisodeRegistry,
    EpisodeRuntime,
    keyword_coverage,
    normalize_text,
    priority_distance,
    tag_f1,
    tokenize,
)
from models import EmailTriageAction  # noqa: E402


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_punctuation_removed(self):
        result = normalize_text("Hello, World!")
        assert "," not in result
        assert "!" not in result

    def test_empty(self):
        assert normalize_text("") == ""


class TestTokenize:
    def test_basic_split(self):
        tokens = tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_strips_punctuation(self):
        tokens = tokenize("Hello, World!")
        assert "hello" in tokens

    def test_empty(self):
        assert tokenize("") == []


class TestKeywordCoverage:
    def test_full_coverage(self):
        score = keyword_coverage("I need a refund for my payment", ["refund", "payment"])
        assert score == pytest.approx(1.0)

    def test_partial_coverage(self):
        score = keyword_coverage("I need a refund", ["refund", "payment"])
        assert 0.0 < score < 1.0

    def test_no_match(self):
        score = keyword_coverage("nothing relevant here", ["refund", "payment"])
        assert score == pytest.approx(0.0)

    def test_empty_keywords(self):
        assert keyword_coverage("any text", []) == pytest.approx(0.0)

    def test_substring_match(self):
        # "apologize" should match even when text contains "apologizes"
        score = keyword_coverage("I sincerely apologizes for this", ["apologize"])
        assert score == pytest.approx(1.0)


class TestPriorityDistance:
    def test_same(self):
        assert priority_distance("High", "High") == 0

    def test_adjacent(self):
        assert priority_distance("High", "Medium") == 1
        assert priority_distance("Critical", "High") == 1

    def test_two_apart(self):
        assert priority_distance("Critical", "Medium") == 2

    def test_max_distance(self):
        assert priority_distance("Critical", "Low") == 3


class TestTagF1:
    def test_perfect_match(self):
        assert tag_f1(["complaint", "refund"], ["complaint", "refund"]) == pytest.approx(1.0)

    def test_empty_pred_nonempty_true(self):
        assert tag_f1([], ["complaint"]) == pytest.approx(0.0)

    def test_empty_both(self):
        assert tag_f1([], []) == pytest.approx(1.0)

    def test_partial_overlap(self):
        f1 = tag_f1(["complaint", "inquiry"], ["complaint", "refund"])
        assert 0.0 < f1 < 1.0

    def test_no_overlap(self):
        f1 = tag_f1(["feedback"], ["billing-dispute"])
        assert f1 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EmailTaskCatalog
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def catalog() -> EmailTaskCatalog:
    return EmailTaskCatalog()


class TestEmailTaskCatalog:
    def test_loads_emails(self, catalog: EmailTaskCatalog):
        assert catalog.get_email("e001") is not None
        assert catalog.get_email("e007") is not None

    def test_email_has_ground_truth(self, catalog: EmailTaskCatalog):
        email = catalog.get_email("e001")
        assert email.ground_truth.department == "Billing"
        assert email.ground_truth.priority == "High"
        assert email.ground_truth.is_spam is False

    def test_spam_email(self, catalog: EmailTaskCatalog):
        spam = catalog.get_email("e007")
        assert spam.ground_truth.is_spam is True

    def test_critical_email(self, catalog: EmailTaskCatalog):
        critical = catalog.get_email("e006")
        assert critical.ground_truth.priority == "Critical"
        assert critical.ground_truth.should_escalate is True

    def test_loads_three_tasks(self, catalog: EmailTaskCatalog):
        tasks = catalog.list_tasks()
        assert len(tasks) == 3

    def test_task_order(self, catalog: EmailTaskCatalog):
        task_ids = [t.task_id for t in catalog.list_tasks()]
        assert task_ids == ["routing_easy", "priority_medium", "drafting_hard"]

    def test_routing_easy_emails(self, catalog: EmailTaskCatalog):
        task = catalog.get_task("routing_easy")
        assert len(task.email_ids) == 5

    def test_priority_medium_emails(self, catalog: EmailTaskCatalog):
        task = catalog.get_task("priority_medium")
        assert len(task.email_ids) == 6

    def test_drafting_hard_emails(self, catalog: EmailTaskCatalog):
        task = catalog.get_task("drafting_hard")
        assert len(task.email_ids) == 4

    def test_unknown_email_returns_none(self, catalog: EmailTaskCatalog):
        assert catalog.get_email("not-an-email") is None

    def test_unknown_task_returns_none(self, catalog: EmailTaskCatalog):
        assert catalog.get_task("not-a-task") is None

    def test_tasks_response(self, catalog: EmailTaskCatalog):
        resp = catalog.build_tasks_response()
        assert len(resp.tasks) == 3
        assert "action_schema" in resp.model_dump()
        assert "observation_schema" in resp.model_dump()


# ---------------------------------------------------------------------------
# EpisodeRuntime — action processing
# ---------------------------------------------------------------------------


@pytest.fixture
def runtime_easy(catalog: EmailTaskCatalog) -> EpisodeRuntime:
    return EpisodeRuntime(catalog=catalog, task_id="routing_easy")


@pytest.fixture
def runtime_medium(catalog: EmailTaskCatalog) -> EpisodeRuntime:
    return EpisodeRuntime(catalog=catalog, task_id="priority_medium")


@pytest.fixture
def runtime_hard(catalog: EmailTaskCatalog) -> EpisodeRuntime:
    return EpisodeRuntime(catalog=catalog, task_id="drafting_hard")


class TestReadEmail:
    def test_read_gives_small_reward(self, runtime_easy: EpisodeRuntime):
        reward, bonuses, _ = runtime_easy.apply_action(
            EmailTriageAction(tool="read_email", email_id="e001")
        )
        assert reward > 0
        assert "email_read" in bonuses

    def test_read_sets_active_email(self, runtime_easy: EpisodeRuntime):
        runtime_easy.apply_action(
            EmailTriageAction(tool="read_email", email_id="e001")
        )
        assert runtime_easy._active_email_id == "e001"

    def test_read_marks_email_as_read(self, runtime_easy: EpisodeRuntime):
        runtime_easy.apply_action(
            EmailTriageAction(tool="read_email", email_id="e001")
        )
        assert runtime_easy.progress["e001"].has_been_read is True


class TestClassifyEmail:
    def test_correct_routing_reward(self, runtime_easy: EpisodeRuntime):
        runtime_easy.apply_action(
            EmailTriageAction(tool="read_email", email_id="e001")
        )
        reward, bonuses, _ = runtime_easy.apply_action(
            EmailTriageAction(
                tool="classify_email", email_id="e001", department="Billing"
            )
        )
        assert reward > 0
        assert "correct_department" in bonuses
        assert bonuses["correct_department"] == pytest.approx(0.20)

    def test_wrong_routing_penalty(self, runtime_easy: EpisodeRuntime):
        runtime_easy.apply_action(
            EmailTriageAction(tool="read_email", email_id="e001")
        )
        reward, _, penalties = runtime_easy.apply_action(
            EmailTriageAction(
                tool="classify_email", email_id="e001", department="Sales"
            )
        )
        assert reward < 0
        assert "wrong_department" in penalties

    def test_classify_without_read_penalty(self, runtime_easy: EpisodeRuntime):
        _, _, penalties = runtime_easy.apply_action(
            EmailTriageAction(
                tool="classify_email", email_id="e001", department="Billing"
            )
        )
        assert "classify_without_read" in penalties


class TestSetPriority:
    def test_correct_priority_reward(self, runtime_medium: EpisodeRuntime):
        runtime_medium.apply_action(
            EmailTriageAction(tool="read_email", email_id="e006")
        )
        reward, bonuses, _ = runtime_medium.apply_action(
            EmailTriageAction(
                tool="set_priority", email_id="e006", priority="Critical"
            )
        )
        assert "correct_priority" in bonuses
        assert reward > 0

    def test_adjacent_priority_partial_reward(self, runtime_medium: EpisodeRuntime):
        runtime_medium.apply_action(
            EmailTriageAction(tool="read_email", email_id="e008")
        )
        # e008 is High; assigning Medium is adjacent
        reward, bonuses, _ = runtime_medium.apply_action(
            EmailTriageAction(
                tool="set_priority", email_id="e008", priority="Medium"
            )
        )
        assert "adjacent_priority" in bonuses
        assert 0.0 < reward

    def test_far_priority_penalty(self, runtime_medium: EpisodeRuntime):
        runtime_medium.apply_action(
            EmailTriageAction(tool="read_email", email_id="e006")
        )
        # e006 is Critical; assigning Low is 3 tiers away
        reward, _, penalties = runtime_medium.apply_action(
            EmailTriageAction(
                tool="set_priority", email_id="e006", priority="Low"
            )
        )
        assert "wrong_priority" in penalties
        assert reward < 0

    def test_sla_timing_bonus_early(self, runtime_medium: EpisodeRuntime):
        runtime_medium.step_count = 5  # early in episode
        runtime_medium.apply_action(
            EmailTriageAction(tool="read_email", email_id="e006")
        )
        _, bonuses, _ = runtime_medium.apply_action(
            EmailTriageAction(
                tool="set_priority", email_id="e006", priority="Critical"
            )
        )
        assert "sla_timing" in bonuses


class TestAddTags:
    def test_correct_tag_reward(self, runtime_medium: EpisodeRuntime):
        runtime_medium.apply_action(
            EmailTriageAction(tool="read_email", email_id="e008")
        )
        reward, bonuses, _ = runtime_medium.apply_action(
            EmailTriageAction(
                tool="add_tags", email_id="e008", tags=["complaint"]
            )
        )
        assert "correct_tag_complaint" in bonuses
        assert reward > 0

    def test_wrong_tag_penalty(self, runtime_medium: EpisodeRuntime):
        runtime_medium.apply_action(
            EmailTriageAction(tool="read_email", email_id="e008")
        )
        # "feature-request" is not in e008's ground truth tags
        _, _, penalties = runtime_medium.apply_action(
            EmailTriageAction(
                tool="add_tags", email_id="e008", tags=["feature-request"]
            )
        )
        assert any("wrong_tag" in k for k in penalties)


class TestSpamDetection:
    def test_correct_spam_reward(self, runtime_medium: EpisodeRuntime):
        runtime_medium.apply_action(
            EmailTriageAction(tool="read_email", email_id="e007")
        )
        reward, bonuses, _ = runtime_medium.apply_action(
            EmailTriageAction(tool="mark_spam", email_id="e007")
        )
        assert "correct_spam_detection" in bonuses
        assert reward > 0

    def test_false_positive_penalty(self, runtime_easy: EpisodeRuntime):
        runtime_easy.apply_action(
            EmailTriageAction(tool="read_email", email_id="e001")
        )
        # e001 is NOT spam
        reward, _, penalties = runtime_easy.apply_action(
            EmailTriageAction(tool="mark_spam", email_id="e001")
        )
        assert "false_spam_positive" in penalties
        assert reward < 0


class TestDraftResponse:
    def test_response_with_keywords_rewards(self, runtime_hard: EpisodeRuntime):
        runtime_hard.apply_action(
            EmailTriageAction(tool="read_email", email_id="e012")
        )
        reward, bonuses, _ = runtime_hard.apply_action(
            EmailTriageAction(
                tool="draft_response",
                email_id="e012",
                response_text=(
                    "Dear Amanda, I sincerely apologize for the repeated billing errors. "
                    "I will investigate your account immediately and ensure a full refund "
                    "of $350 and a credit are issued within 24 hours to resolve this. Best regards."
                ),
            )
        )
        assert "response_quality" in bonuses
        assert bonuses["response_quality"] > 0

    def test_professional_tone_bonus(self, runtime_hard: EpisodeRuntime):
        runtime_hard.apply_action(
            EmailTriageAction(tool="read_email", email_id="e012")
        )
        _, bonuses, _ = runtime_hard.apply_action(
            EmailTriageAction(
                tool="draft_response",
                email_id="e012",
                response_text=(
                    "Dear Amanda, I apologize and will investigate. "
                    "Thank you for your patience. Best regards."
                ),
            )
        )
        assert "professional_tone" in bonuses

    def test_unprofessional_tone_penalty(self, runtime_hard: EpisodeRuntime):
        runtime_hard.apply_action(
            EmailTriageAction(tool="read_email", email_id="e012")
        )
        _, _, penalties = runtime_hard.apply_action(
            EmailTriageAction(
                tool="draft_response",
                email_id="e012",
                response_text="We will look into it.",
            )
        )
        assert "unprofessional_tone" in penalties


class TestDuplicateAction:
    def test_duplicate_read_penalized(self, runtime_easy: EpisodeRuntime):
        runtime_easy.apply_action(
            EmailTriageAction(tool="read_email", email_id="e001")
        )
        reward, _, penalties = runtime_easy.apply_action(
            EmailTriageAction(tool="read_email", email_id="e001")
        )
        assert "duplicate_action" in penalties
        assert reward == pytest.approx(GLOBAL_DUPLICATE_ACTION_PENALTY)

    def test_duplicate_classify_penalized(self, runtime_easy: EpisodeRuntime):
        runtime_easy.apply_action(
            EmailTriageAction(tool="read_email", email_id="e001")
        )
        runtime_easy.apply_action(
            EmailTriageAction(
                tool="classify_email", email_id="e001", department="Billing"
            )
        )
        reward, _, penalties = runtime_easy.apply_action(
            EmailTriageAction(
                tool="classify_email", email_id="e001", department="Billing"
            )
        )
        assert "duplicate_action" in penalties


class TestInvalidEmailId:
    def test_invalid_email_id_penalty(self, runtime_easy: EpisodeRuntime):
        reward, _, penalties = runtime_easy.apply_action(
            EmailTriageAction(tool="read_email", email_id="e999")
        )
        assert "invalid_email_id" in penalties
        assert reward < 0


# ---------------------------------------------------------------------------
# Scoring formulae
# ---------------------------------------------------------------------------


class TestRoutingEasyScore:
    def test_zero_score_no_actions(self, catalog: EmailTaskCatalog):
        runtime = EpisodeRuntime(catalog=catalog, task_id="routing_easy")
        report = runtime.compute_grader_report()
        assert report.score == pytest.approx(0.0)

    def test_perfect_score(self, catalog: EmailTaskCatalog):
        runtime = EpisodeRuntime(catalog=catalog, task_id="routing_easy")
        task = catalog.get_task("routing_easy")
        for eid in task.email_ids:
            rec = catalog.get_email(eid)
            runtime.apply_action(EmailTriageAction(tool="read_email", email_id=eid))
            runtime.apply_action(
                EmailTriageAction(
                    tool="classify_email",
                    email_id=eid,
                    department=rec.ground_truth.department,
                )
            )
        report = runtime.compute_grader_report()
        assert report.score == pytest.approx(1.0)

    def test_partial_score(self, catalog: EmailTaskCatalog):
        runtime = EpisodeRuntime(catalog=catalog, task_id="routing_easy")
        # Correctly classify only e001
        runtime.apply_action(EmailTriageAction(tool="read_email", email_id="e001"))
        runtime.apply_action(
            EmailTriageAction(
                tool="classify_email", email_id="e001", department="Billing"
            )
        )
        report = runtime.compute_grader_report()
        assert 0.0 < report.score < 1.0


class TestPriorityMediumScore:
    def test_spam_detection_contributes(self, catalog: EmailTaskCatalog):
        runtime = EpisodeRuntime(catalog=catalog, task_id="priority_medium")
        runtime.apply_action(EmailTriageAction(tool="read_email", email_id="e007"))
        runtime.apply_action(EmailTriageAction(tool="mark_spam", email_id="e007"))
        report = runtime.compute_grader_report()
        assert report.score > 0.0

    def test_combined_score_components(self, catalog: EmailTaskCatalog):
        runtime = EpisodeRuntime(catalog=catalog, task_id="priority_medium")
        task = catalog.get_task("priority_medium")
        for eid in task.email_ids:
            rec = catalog.get_email(eid)
            runtime.apply_action(EmailTriageAction(tool="read_email", email_id=eid))
            if rec.ground_truth.is_spam:
                runtime.apply_action(EmailTriageAction(tool="mark_spam", email_id=eid))
            else:
                runtime.apply_action(
                    EmailTriageAction(
                        tool="classify_email",
                        email_id=eid,
                        department=rec.ground_truth.department,
                    )
                )
                runtime.apply_action(
                    EmailTriageAction(
                        tool="set_priority",
                        email_id=eid,
                        priority=rec.ground_truth.priority,
                    )
                )
                if rec.ground_truth.tags:
                    runtime.apply_action(
                        EmailTriageAction(
                            tool="add_tags",
                            email_id=eid,
                            tags=rec.ground_truth.tags[:3],
                        )
                    )
        report = runtime.compute_grader_report()
        # Perfect actions should yield a high score
        assert report.score > 0.7


class TestDraftingHardScore:
    def test_zero_score_no_actions(self, catalog: EmailTaskCatalog):
        runtime = EpisodeRuntime(catalog=catalog, task_id="drafting_hard")
        report = runtime.compute_grader_report()
        assert report.score == pytest.approx(0.0)

    def test_score_range(self, catalog: EmailTaskCatalog):
        runtime = EpisodeRuntime(catalog=catalog, task_id="drafting_hard")
        report = runtime.compute_grader_report()
        assert 0.0 <= report.score <= 1.0


# ---------------------------------------------------------------------------
# IsDone
# ---------------------------------------------------------------------------


class TestIsDone:
    def test_done_when_budget_exhausted(self, catalog: EmailTaskCatalog):
        runtime = EpisodeRuntime(catalog=catalog, task_id="routing_easy")
        runtime.step_count = runtime.task.step_budget
        assert runtime.is_done() is True

    def test_done_when_all_processed(self, catalog: EmailTaskCatalog):
        runtime = EpisodeRuntime(catalog=catalog, task_id="routing_easy")
        for ep in runtime.progress.values():
            ep.status = "classified"
        assert runtime.is_done() is True

    def test_not_done_initially(self, catalog: EmailTaskCatalog):
        runtime = EpisodeRuntime(catalog=catalog, task_id="routing_easy")
        assert runtime.is_done() is False


# ---------------------------------------------------------------------------
# GraderReport structure
# ---------------------------------------------------------------------------


class TestGraderReport:
    def test_report_fields(self, catalog: EmailTaskCatalog):
        runtime = EpisodeRuntime(catalog=catalog, task_id="routing_easy")
        report = runtime.compute_grader_report()
        assert report.episode_id == runtime.episode_id
        assert report.task_id == "routing_easy"
        assert report.difficulty == "easy"
        assert 0.0 <= report.score <= 1.0
        assert len(report.email_grades) == 5

    def test_email_grade_fields(self, catalog: EmailTaskCatalog):
        runtime = EpisodeRuntime(catalog=catalog, task_id="routing_easy")
        runtime.apply_action(EmailTriageAction(tool="read_email", email_id="e001"))
        runtime.apply_action(
            EmailTriageAction(
                tool="classify_email", email_id="e001", department="Billing"
            )
        )
        report = runtime.compute_grader_report()
        grade = next(g for g in report.email_grades if g.email_id == "e001")
        assert grade.score == pytest.approx(1.0)
        assert "routing" in grade.component_scores


# ---------------------------------------------------------------------------
# EpisodeRegistry
# ---------------------------------------------------------------------------


class TestEpisodeRegistry:
    def test_store_and_retrieve(self, catalog: EmailTaskCatalog):
        registry = EpisodeRegistry()
        runtime = EpisodeRuntime(catalog=catalog, task_id="routing_easy")
        report = runtime.compute_grader_report()
        registry.store(report)
        retrieved = registry.get(report.episode_id)
        assert retrieved is not None
        assert retrieved.episode_id == report.episode_id

    def test_missing_episode_returns_none(self):
        registry = EpisodeRegistry()
        assert registry.get("does-not-exist") is None

    def test_overwrite_is_idempotent(self, catalog: EmailTaskCatalog):
        registry = EpisodeRegistry()
        runtime = EpisodeRuntime(catalog=catalog, task_id="routing_easy")
        report = runtime.compute_grader_report()
        registry.store(report)
        registry.store(report)  # second store should not raise
        assert registry.get(report.episode_id) is not None
