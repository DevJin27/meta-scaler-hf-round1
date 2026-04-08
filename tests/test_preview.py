"""Unit tests for the ad-hoc preview triage heuristics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline import scripted_triage_preview  # noqa: E402


class TestScriptedTriagePreview:
    def test_billing_dispute_human_email_gets_reply(self):
        result = scripted_triage_preview(
            sender="alex.customer@example.com",
            subject="Incorrect charge on my invoice",
            body=(
                "Hi team, I was charged twice on my latest invoice and need "
                "help with a refund. This is the third time this has happened."
            ),
        )
        assert result.spam is False
        assert result.department == "Billing"
        assert result.priority == "High"
        assert "refund" in result.tags
        assert "billing-dispute" in result.tags
        assert "draft_response" in result.suggested_actions
        assert result.response_text is not None

    def test_footer_noise_does_not_turn_human_bug_report_into_inquiry(self):
        result = scripted_triage_preview(
            sender="marco.bianchi@example.com",
            subject="Desktop app crashes when I upload files over 10MB",
            body=(
                "Hi Support,\n\n"
                "The application crashes every time I try to upload a file larger "
                "than 10MB. This started after the latest update and blocks my daily work.\n\n"
                "If you have any questions or concerns, contact support@example.com.\n"
                "Privacy Policy"
            ),
        )
        assert result.department == "Technical Support"
        assert result.priority == "High"
        assert "technical-issue" in result.tags
        assert result.response_text is not None

    def test_verification_code_email_is_account_notification_with_no_reply(self):
        result = scripted_triage_preview(
            sender="no-reply@notify.docker.com",
            subject="[Docker] Complete your account creation - one-time verification code",
            body=(
                "Hello isodevv!\n\n"
                "Thanks for joining Docker. Your verification code is: 0KCJ8D.\n"
                "Navigate back to your browser and enter the code. This code will expire in 15 minutes.\n\n"
                "If you have any questions or concerns, please contact us at support@docker.com.\n"
                "This email was sent to notify you of an update that was made to your Docker Account."
            ),
        )
        assert result.spam is False
        assert result.department == "Account Management"
        assert result.priority == "Low"
        assert result.response_text is None
        assert "draft_response" not in result.suggested_actions

    def test_billing_receipt_is_low_priority_notification(self):
        result = scripted_triage_preview(
            sender="receipts@billing.vendor.example",
            subject="Your receipt for March 2026",
            body=(
                "Thanks for your payment. Your receipt is attached and your subscription "
                "has been renewed for another month."
            ),
        )
        assert result.department == "Billing"
        assert result.priority == "Low"
        assert result.response_text is None

    def test_order_update_is_low_priority_account_notification(self):
        result = scripted_triage_preview(
            sender="updates@shop.example",
            subject="Your order has shipped",
            body=(
                "Good news. Your package has shipped and is out for delivery tomorrow. "
                "Use the tracking number in your account for status updates."
            ),
        )
        assert result.department == "Account Management"
        assert result.priority == "Low"
        assert "order-status" in result.tags
        assert result.response_text is None

    def test_marketing_newsletter_is_sales_without_reply(self):
        result = scripted_triage_preview(
            sender="updates@product.example",
            subject="Spring product roundup",
            body=(
                "Check out our new features and roadmap update for this quarter.\n"
                "Manage preferences or unsubscribe at any time."
            ),
        )
        assert result.department == "Sales"
        assert result.priority == "Low"
        assert result.response_text is None
        assert "draft_response" not in result.suggested_actions

    def test_status_alert_is_high_priority_and_one_way(self):
        result = scripted_triage_preview(
            sender="status@infra.example",
            subject="Incident update: degraded performance in API region",
            body=(
                "We are investigating degraded performance affecting API requests. "
                "A follow-up status update will be sent in 30 minutes."
            ),
        )
        assert result.department == "Technical Support"
        assert result.priority == "High"
        assert "service-outage" in result.tags
        assert result.response_text is None

    def test_security_alert_is_high_priority_and_one_way(self):
        result = scripted_triage_preview(
            sender="no-reply@github.example",
            subject="Security alert: new sign-in to your account",
            body=(
                "We noticed a new sign-in to your account from a new device. "
                "If this was not you, reset your password immediately."
            ),
        )
        assert result.department == "Technical Support"
        assert result.priority == "High"
        assert result.response_text is None

    def test_calendar_invite_is_low_priority_with_no_reply(self):
        result = scripted_triage_preview(
            sender="arina.ali2024@nst.rishihood.edu.in",
            subject=(
                "Invitation: Neutron <> Competitions Team - Bandwidth 4 "
                "@ Mon 2026-03-30 6:30pm - 8pm (IST) (Dev Jindal)"
            ),
            body=(
                "View on Google Calendar\n"
                "When Mon 2026-03-30 6:30pm - 8pm (IST)\n"
                "Agenda: Confirm the competitions and finalise documents required for them.\n"
                "Reply for dev.jindal2024@nst.rishihood.edu.in\n"
                "Yes\nNo\nMaybe\n"
                "Invitation from Google Calendar"
            ),
        )
        assert result.department == "General"
        assert result.priority == "Low"
        assert result.response_text is None
        assert "draft_response" not in result.suggested_actions

    def test_generic_automated_account_notice_stays_one_way(self):
        result = scripted_triage_preview(
            sender="no-reply@service.example",
            subject="Your account preferences were updated",
            body=(
                "This is a confirmation that your account preferences were updated successfully. "
                "No further action is required."
            ),
        )
        assert result.department == "Account Management"
        assert result.priority == "Low"
        assert result.response_text is None
        assert "draft_response" not in result.suggested_actions

    def test_obvious_spam_still_gets_marked_spam(self):
        result = scripted_triage_preview(
            sender="promo@totally-legit.example",
            subject="Congratulations winner",
            body=(
                "Claim your prize now. Click here for free money and send your "
                "bank account details to receive your winnings."
            ),
        )
        assert result.spam is True
        assert result.suggested_actions == ["read_email", "mark_spam"]
        assert result.response_text is None
