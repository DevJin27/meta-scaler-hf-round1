"""Scripted and LLM baseline runners for the EmailTriage benchmark.

Scripted baseline (no API key required)
    Uses keyword matching for routing, urgency detection for priority,
    and templated responses for drafting.  Produces reproducible scores.

LLM baseline (requires OPENAI_API_KEY)
    Drives the environment via the OpenAI chat-completions API using a
    zero-shot system prompt.  Compatible with any OpenAI-compatible endpoint.

Expected scripted baseline scores
    routing_easy:    ~0.60
    priority_medium: ~0.50
    drafting_hard:   ~0.38
    overall:         ~0.49
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional

try:
    from emailtriage.email_core import TASK_ORDER
    from emailtriage.models import (
        BaselineReport,
        BaselineTaskResult,
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriagePreviewResponse,
        GraderReport,
    )
except ImportError:
    from email_core import TASK_ORDER
    from models import (
        BaselineReport,
        BaselineTaskResult,
        EmailTriageAction,
        EmailTriageObservation,
        EmailTriagePreviewResponse,
        GraderReport,
    )

DEFAULT_BASELINE_MODEL: str = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Scripted routing knowledge
# ---------------------------------------------------------------------------

_DEPT_KEYWORDS: Dict[str, List[str]] = {
    "Billing": [
        "invoice", "charge", "billing", "payment", "refund", "bill",
        "subscription", "fee", "cost", "overcharge", "credit card",
        "debit", "transaction", "price", "amount", "account charge",
    ],
    "Technical Support": [
        "error", "bug", "crash", "not working", "broken", "login",
        "password", "reset", "access denied", "outage", "down",
        "issue", "problem", "api", "integration", "server", "database",
        "hacked", "compromised", "security", "timeout", "connection",
    ],
    "Account Management": [
        "order", "shipping", "delivery", "tracking", "address",
        "cancel", "close account", "closure", "account update",
        "profile", "subscription cancel", "account settings",
    ],
    "Sales": [
        "pricing", "plan", "enterprise", "upgrade", "demo",
        "quote", "purchase", "license", "partner", "partnership",
        "white-label", "procurement", "contract",
    ],
    "HR": [
        "employee", "benefits", "payroll", "vacation", "leave",
        "hr", "human resources", "salary", "enrollment", "401k",
        "health insurance", "onboarding",
    ],
}

_SPAM_INDICATORS: List[str] = [
    "winner", "won", "lottery", "prize", "congratulations",
    "million dollars", "claim", "free money", "free gift",
    "pre-approved", "gift card", "click here", "verify your identity",
    "bank account details", "limited time",
]

_HARD_SPAM_KEYWORDS: List[str] = [
    "lottery", "million dollars", "bank account details", "wire transfer",
    "claim your prize",
]

_AUTOMATED_SENDER_HINTS: List[str] = [
    "no-reply", "noreply", "do-not-reply", "donotreply", "notify",
    "notification", "notifications", "mailer-daemon",
]

_ACCOUNT_NOTIFICATION_KEYWORDS: List[str] = [
    "verification code", "one-time verification code", "one-time code",
    "passcode", "otp", "security code", "sign in code", "sign-in code",
    "login code", "verification email", "verify your email",
    "confirm your email", "account creation", "create your account",
    "password reset", "reset your password", "reset code",
    "authentication code", "two-factor", "2fa",
]

_BILLING_NOTIFICATION_KEYWORDS: List[str] = [
    "receipt", "invoice attached", "invoice is ready", "invoice available",
    "payment received", "payment confirmation", "subscription renewed",
    "subscription renewal", "billing statement", "charge receipt",
    "transaction receipt", "failed payment", "payment failed",
]

_ORDER_NOTIFICATION_KEYWORDS: List[str] = [
    "order confirmation", "tracking number", "shipment", "shipped",
    "out for delivery", "delivery update", "delivered",
]

_CALENDAR_INVITE_KEYWORDS: List[str] = [
    "invitation from google calendar",
    "view on google calendar",
    "you are receiving this email because you are subscribed to calendar notifications",
    "reply yes",
    "reply no",
    "reply maybe",
]

_MARKETING_KEYWORDS: List[str] = [
    "unsubscribe", "manage preferences", "view in browser", "newsletter",
    "product update", "special offer", "limited time offer",
    "limited-time offer", "% off", "promotion", "promo code",
    "new features", "feature roundup",
]

_SECURITY_ALERT_KEYWORDS: List[str] = [
    "suspicious login", "new login", "new sign-in", "sign-in attempt",
    "login attempt", "security alert", "unrecognized device",
    "password changed", "account compromised",
]

_STATUS_ALERT_KEYWORDS: List[str] = [
    "service outage", "degraded performance", "incident update",
    "resolved incident", "maintenance window",
]

_BOILERPLATE_LINE_KEYWORDS: List[str] = [
    "if you have any questions or concerns",
    "this email was sent to",
    "you are receiving this email",
    "privacy policy",
    "manage preferences",
    "unsubscribe",
    "view in browser",
    "all rights reserved",
]

_PRIORITY_KEYWORDS: Dict[str, List[str]] = {
    "Critical": [
        "urgent", "critical", "production down", "system down",
        "database down", "compromised", "hacked", "breach",
        "emergency", "asap", "immediately", "outage",
        "cannot process", "all customers affected",
    ],
    "High": [
        "important", "high priority", "vip", "business impact",
        "payment failed", "failed", "frustrated", "angry",
        "final warning", "chargeback", "revenue", "enterprise",
        "third time", "repeated", "incorrect charge",
    ],
    "Medium": [
        "question", "inquiry", "issue", "problem", "wondering",
        "need assistance", "clarification", "discrepancy",
    ],
    "Low": [
        "suggestion", "feedback", "feature request", "when possible",
        "just wanted", "wondering", "feature", "dark mode",
    ],
}

_RESPONSE_TEMPLATES: Dict[str, str] = {
    "Billing": (
        "Dear {name},\n\n"
        "Thank you for contacting us. I sincerely apologize for the billing issue "
        "you have experienced.\n\n"
        "I have escalated this to our billing team and will investigate your account "
        "immediately. We will resolve this and process any necessary refund or credit "
        "within 24 hours.\n\n"
        "Please don't hesitate to reach out if you need further assistance.\n\n"
        "Best regards,\nCustomer Support Team"
    ),
    "Technical Support": (
        "Dear {name},\n\n"
        "Thank you for reaching out. I apologize for the technical difficulties "
        "you are experiencing.\n\n"
        "I have escalated this to our engineering team who will investigate your "
        "issue immediately and provide logs analysis and a resolution update within "
        "4 hours.\n\n"
        "Best regards,\nTechnical Support Team"
    ),
    "Account Management": (
        "Dear {name},\n\n"
        "Thank you for contacting us. I understand your request and will process it "
        "promptly.\n\n"
        "We have noted the details and will confirm the updates to your account. "
        "Please allow 1 business day for changes to take effect.\n\n"
        "Sincerely,\nAccount Management Team"
    ),
    "Sales": (
        "Dear {name},\n\n"
        "Thank you for your interest. I would be happy to discuss our plans and "
        "pricing with you.\n\n"
        "A member of our sales team will contact you within 1 business day to arrange "
        "a demo and discuss your requirements.\n\n"
        "Best regards,\nSales Team"
    ),
    "General": (
        "Dear {name},\n\n"
        "Thank you for reaching out. We appreciate your feedback.\n\n"
        "We will review your message and get back to you within 1-2 business days.\n\n"
        "Best regards,\nCustomer Support Team"
    ),
}

# ---------------------------------------------------------------------------
# Scripted classifier functions
# ---------------------------------------------------------------------------


def _score_department(text: str) -> str:
    """Keyword-frequency based department classifier."""
    low = text.lower()
    scores: Dict[str, int] = {}
    for dept, kws in _DEPT_KEYWORDS.items():
        scores[dept] = sum(1 for kw in kws if kw in low)
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "General"


def _score_priority(text: str, department: str) -> str:
    """Urgency-keyword based priority classifier."""
    low = text.lower()
    for level in ("Critical", "High", "Medium", "Low"):
        if any(kw in low for kw in _PRIORITY_KEYWORDS[level]):
            return level
    # Department-based defaults
    if department in ("Technical Support",):
        return "High"
    if department in ("Billing",):
        return "Medium"
    return "Low"


def _detect_spam(text: str) -> bool:
    """Return True if text contains 2+ spam signals."""
    low = text.lower()
    hits = sum(1 for kw in _SPAM_INDICATORS if kw in low)
    return hits >= 2


def _select_tags(text: str) -> List[str]:
    """Assign up to 3 tags using keyword heuristics."""
    low = text.lower()
    tag_map: Dict[str, List[str]] = {
        "complaint": ["frustrated", "angry", "unacceptable", "terrible", "complaint", "final warning"],
        "inquiry": ["question", "wondering", "how to", "what is", "information", "clarify"],
        "refund": ["refund", "money back", "return payment"],
        "password-reset": ["password", "reset password", "forgot password"],
        "order-status": ["order", "tracking", "delivery", "shipping", "where is my order"],
        "technical-issue": ["error", "crash", "bug", "not working", "broken", "timeout"],
        "account-closure": ["close account", "closure", "cancel account"],
        "payment-failed": ["payment failed", "declined", "transaction failed"],
        "billing-dispute": ["wrong charge", "incorrect", "overcharged", "billing error", "discrepancy"],
        "service-outage": ["down", "outage", "unavailable", "offline", "system down"],
        "feature-request": ["feature", "dark mode", "suggest", "would be nice"],
        "feedback": ["feedback", "suggestion", "improve", "love the app"],
    }
    tags: List[str] = []
    for tag, kws in tag_map.items():
        if any(kw in low for kw in kws):
            tags.append(tag)
        if len(tags) == 3:
            break
    return tags


def _matched_keywords(text: str, keywords: List[str]) -> List[str]:
    """Return matched keywords in the order they appear in *keywords*."""
    low = text.lower()
    return [kw for kw in keywords if kw in low]


def _sender_display_name(sender: str) -> str:
    """Convert an email address into a readable customer name."""
    local_part = sender.split("@", 1)[0]
    cleaned = local_part.replace(".", " ").replace("_", " ").strip()
    return " ".join(part.capitalize() for part in cleaned.split()) or "Valued Customer"


def _contains_any(text: str, keywords: List[str]) -> bool:
    """Return True if any keyword appears in *text*."""
    low = text.lower()
    return any(keyword in low for keyword in keywords)


def _is_automated_sender(sender: str) -> bool:
    """Heuristic for one-way system senders."""
    local_part = sender.split("@", 1)[0].lower()
    return any(hint in local_part for hint in _AUTOMATED_SENDER_HINTS)


def _strip_preview_boilerplate(body: str) -> str:
    """Remove common footer/disclaimer lines before preview classification."""
    kept_lines: List[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low = line.lower()
        if any(keyword in low for keyword in _BOILERPLATE_LINE_KEYWORDS):
            continue
        if low.startswith("©") or low.startswith("copyright"):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _build_preview_text(subject: str, body: str) -> str:
    """Build the text used for preview heuristics after footer cleanup."""
    cleaned_body = _strip_preview_boilerplate(body)
    return f"{subject} {cleaned_body}".strip()


def _preview_notification_kind(sender: str, subject: str, body: str) -> Optional[str]:
    """Classify a non-spam message into a preview-only special bucket."""
    text = f"{subject} {body}"

    if _contains_any(text, _SECURITY_ALERT_KEYWORDS):
        return "security_alert"
    if subject.lower().startswith("invitation:") or _contains_any(text, _CALENDAR_INVITE_KEYWORDS):
        return "calendar_invite"
    if _contains_any(text, _ACCOUNT_NOTIFICATION_KEYWORDS):
        return "account_notification"
    if _contains_any(text, _BILLING_NOTIFICATION_KEYWORDS):
        return "billing_notification"
    if _contains_any(text, _ORDER_NOTIFICATION_KEYWORDS):
        return "order_notification"
    if _contains_any(text, _STATUS_ALERT_KEYWORDS):
        return "status_alert"
    if _contains_any(text, _MARKETING_KEYWORDS):
        return "marketing"
    if _is_automated_sender(sender) and _contains_any(text, ["verification", "receipt", "invoice", "tracking", "account"]):
        return "auto_generated"
    return None


def _preview_detect_spam(sender: str, subject: str, body: str) -> tuple[bool, List[str]]:
    """Detect obvious spam without confusing newsletters with scams."""
    text = f"{subject} {body}"
    hard_hits = _matched_keywords(text, _HARD_SPAM_KEYWORDS)
    spam_hits = _matched_keywords(text, _SPAM_INDICATORS)
    looks_like_marketing = _contains_any(text, _MARKETING_KEYWORDS)

    if hard_hits:
        return True, hard_hits
    if len(spam_hits) >= 2 and not looks_like_marketing:
        return True, spam_hits
    return False, spam_hits


def _notification_tags(kind: str, text: str) -> List[str]:
    """Return tags for known notification categories."""
    if kind == "account_notification":
        return ["password-reset"] if "password reset" in text.lower() else []
    if kind == "billing_notification":
        return ["payment-failed"] if "payment failed" in text.lower() or "failed payment" in text.lower() else []
    if kind == "order_notification":
        return ["order-status"]
    if kind == "status_alert":
        return ["service-outage"]
    return []


def scripted_triage_preview(
    *,
    sender: str,
    subject: str,
    body: str,
) -> EmailTriagePreviewResponse:
    """Preview how the scripted baseline would triage a single raw email."""
    full_text = f"{subject} {body}"
    preview_text = _build_preview_text(subject, body)
    is_spam, spam_hits = _preview_detect_spam(sender, subject, body)

    if is_spam:
        return EmailTriagePreviewResponse(
            sender=sender,
            subject=subject,
            suggested_actions=["read_email", "mark_spam"],
            spam=True,
            tags=["spam"],
            explanation=[
                f"Matched spam signals: {', '.join(spam_hits[:5])}.",
                "Recommended action: mark as spam and skip drafting a reply.",
            ],
        )

    kind = _preview_notification_kind(sender, subject, body)
    explanation: List[str] = []
    tags: List[str]
    response_text: Optional[str]

    if kind == "security_alert":
        department = "Technical Support"
        priority = "High"
        tags = _notification_tags(kind, full_text)
        response_text = None
        explanation.append(
            "Detected an automated security alert from sign-in or account-protection language."
        )
        explanation.append(
            "Assigned High priority because account-security events should be reviewed quickly."
        )
    elif kind == "status_alert":
        department = "Technical Support"
        priority = "High"
        tags = _notification_tags(kind, full_text)
        response_text = None
        explanation.append(
            "Detected a service-status or outage notification."
        )
        explanation.append(
            "Assigned High priority because outage notices can affect active work, but no reply draft was created because this is a system update."
        )
    elif kind == "account_notification":
        department = "Account Management"
        priority = "Low"
        tags = _notification_tags(kind, full_text)
        response_text = None
        explanation.append(
            "Detected an account-management notification from verification, password, or account-creation language."
        )
        explanation.append(
            "Assigned Low priority because this reads like an informational system email rather than a support request."
        )
    elif kind == "billing_notification":
        department = "Billing"
        priority = (
            "High"
            if _contains_any(full_text, ["payment failed", "failed payment"])
            else "Low"
        )
        tags = _notification_tags(kind, full_text)
        response_text = None
        explanation.append(
            "Detected an automated billing notification from receipt, invoice, or payment language."
        )
        explanation.append(
            f"Assigned {priority} priority because this is a one-way billing update rather than a conversational support email."
        )
    elif kind == "order_notification":
        department = "Account Management"
        priority = "Low"
        tags = _notification_tags(kind, full_text)
        response_text = None
        explanation.append(
            "Detected an order or delivery status notification."
        )
        explanation.append(
            "Assigned Low priority because it looks informational and does not ask for support."
        )
    elif kind == "marketing":
        department = "Sales"
        priority = "Low"
        tags = []
        response_text = None
        explanation.append(
            "Detected a marketing or newsletter email from promotional language such as unsubscribe or product-update links."
        )
        explanation.append(
            "Assigned Low priority and skipped drafting a reply because this is promotional, not a direct support conversation."
        )
    elif kind == "calendar_invite":
        department = "General"
        priority = "Low"
        tags = []
        response_text = None
        explanation.append(
            "Detected a calendar invitation or meeting notification."
        )
        explanation.append(
            "Assigned Low priority and skipped drafting a reply because this is scheduling information, not a support email."
        )
    elif kind == "auto_generated":
        low_text = full_text.lower()
        if _contains_any(low_text, ["invoice", "receipt", "payment", "charge"]):
            department = "Billing"
            tags = []
        elif _contains_any(low_text, ["tracking", "shipment", "delivery"]):
            department = "Account Management"
            tags = ["order-status"]
        elif _contains_any(low_text, ["account", "profile", "preferences"]):
            department = "Account Management"
            tags = []
        else:
            department = "General"
            tags = []
        priority = "Low"
        response_text = None
        explanation.append(
            "Detected an automated system notification from a one-way sender and transactional wording."
        )
        explanation.append(
            f"Assigned Low priority and routed it to {department} because it looks informational rather than conversational."
        )
    else:
        department_hits = {
            dept: _matched_keywords(preview_text, keywords)
            for dept, keywords in _DEPT_KEYWORDS.items()
        }
        department = _score_department(preview_text)
        matched_priority_level = next(
            (
                level
                for level in ("Critical", "High", "Medium", "Low")
                if _matched_keywords(preview_text, _PRIORITY_KEYWORDS[level])
            ),
            None,
        )
        priority = _score_priority(preview_text, department)
        tags = _select_tags(preview_text)

        if department_hits.get(department):
            explanation.append(
                f"Routed to {department} from keywords: {', '.join(department_hits[department][:5])}."
            )
        else:
            explanation.append(
                "No strong department keywords matched, so it falls back to General."
            )

        if matched_priority_level is not None:
            priority_hits = _matched_keywords(preview_text, _PRIORITY_KEYWORDS[matched_priority_level])
            explanation.append(
                f"Assigned {priority} priority from urgency signals: {', '.join(priority_hits[:5])}."
            )
        else:
            explanation.append(
                f"No urgency keywords matched, so it falls back to {priority} based on the routed department."
            )

        if _is_automated_sender(sender):
            response_text = None
            explanation.append(
                "The sender looks automated, so no reply draft was created."
            )
        else:
            template = _RESPONSE_TEMPLATES.get(department, _RESPONSE_TEMPLATES["General"])
            response_text = template.format(name=_sender_display_name(sender))

    if tags:
        explanation.append(
            f"Suggested tags: {', '.join(tags)}."
        )
    else:
        explanation.append(
            "No tag heuristics matched this message."
        )

    if response_text is None:
        explanation.append(
            "No reply draft was created because this email looks non-conversational or one-way."
        )

    suggested_actions: List[str] = ["read_email", "classify_email", "set_priority"]
    if tags:
        suggested_actions.append("add_tags")
    if response_text is not None:
        suggested_actions.append("draft_response")

    return EmailTriagePreviewResponse(
        sender=sender,
        subject=subject,
        suggested_actions=suggested_actions,
        spam=False,
        department=department,
        priority=priority,
        tags=tags,
        response_text=response_text,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Scripted episode runner
# ---------------------------------------------------------------------------


async def _run_scripted_episode(session: Any, task_id: str) -> str:
    """Run a full episode using the scripted baseline. Returns episode_id."""
    result = await session.reset(task_id=task_id)
    obs: EmailTriageObservation = result.observation
    episode_id = obs.episode_id
    task_difficulty = obs.task_difficulty
    email_ids = [e.email_id for e in obs.inbox_summary.emails]

    for email_id in email_ids:
        if result.done:
            break

        # 1. Read email
        result = await session.step(
            EmailTriageAction(tool="read_email", email_id=email_id)
        )
        if result.done:
            break

        obs = result.observation
        tr = obs.last_tool_result
        subject: str = tr.get("subject", "")
        body: str = tr.get("body", "")
        text = f"{subject} {body}"

        # 2. Spam check
        if _detect_spam(text):
            result = await session.step(
                EmailTriageAction(tool="mark_spam", email_id=email_id)
            )
            if result.done:
                break
            continue

        # 3. Classify department (all tasks)
        department = _score_department(text)
        result = await session.step(
            EmailTriageAction(
                tool="classify_email",
                email_id=email_id,
                department=department,
            )
        )
        if result.done:
            break

        # 4. Set priority (medium + hard tasks)
        if task_difficulty in ("medium", "hard"):
            priority = _score_priority(text, department)
            result = await session.step(
                EmailTriageAction(
                    tool="set_priority",
                    email_id=email_id,
                    priority=priority,
                )
            )
            if result.done:
                break

        # 5. Add tags (medium + hard tasks)
        if task_difficulty in ("medium", "hard"):
            tags = _select_tags(text)
            if tags:
                result = await session.step(
                    EmailTriageAction(
                        tool="add_tags",
                        email_id=email_id,
                        tags=tags,
                    )
                )
                if result.done:
                    break

        # 6. Draft response (hard task only)
        if task_difficulty == "hard":
            sender_name = (
                tr.get("sender", "Valued Customer").split("@")[0].replace(".", " ").title()
            )
            template = _RESPONSE_TEMPLATES.get(department, _RESPONSE_TEMPLATES["General"])
            response_text = template.format(name=sender_name)
            result = await session.step(
                EmailTriageAction(
                    tool="draft_response",
                    email_id=email_id,
                    response_text=response_text,
                )
            )
            if result.done:
                break

    return episode_id


# ---------------------------------------------------------------------------
# LLM episode runner
# ---------------------------------------------------------------------------


async def _run_llm_episode(
    session: Any,
    task_id: str,
    model: str,
    api_key: str,
) -> str:
    """Run an LLM-powered episode using an OpenAI-compatible API."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package is required for the LLM baseline. "
            "Install with: pip install openai"
        ) from exc

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    result = await session.reset(task_id=task_id)
    obs: EmailTriageObservation = result.observation
    episode_id = obs.episode_id

    system_prompt = (
        f"You are an expert customer support agent.\n"
        f"Task: {obs.task_objective}\n\n"
        f"Available tools: {', '.join(obs.allowed_tools)}\n\n"
        "For each email in the inbox:\n"
        "  1. Call read_email to see its content.\n"
        "  2. If spam, call mark_spam.\n"
        "  3. Otherwise: classify_email → set_priority → add_tags → draft_response.\n\n"
        "Always respond with a single JSON object that matches the EmailTriageAction schema:\n"
        f"{json.dumps(EmailTriageAction.model_json_schema(), indent=2)}\n\n"
        "Output ONLY the JSON object — no markdown fences, no explanation."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Inbox:\n{json.dumps(obs.inbox_summary.model_dump(), default=str, indent=2)}\n\n"
                "Begin processing emails."
            ),
        },
    ]

    max_steps = obs.remaining_step_budget
    for _ in range(max_steps):
        if result.done:
            break

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=600,
        )
        content: str = response.choices[0].message.content or ""
        messages.append({"role": "assistant", "content": content})

        # Strip optional markdown fences before parsing
        clean = content.strip()
        if clean.startswith("```"):
            clean = "\n".join(clean.split("\n")[1:])
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]

        try:
            action_dict = json.loads(clean)
            action = EmailTriageAction.model_validate(action_dict)
            result = await session.step(action)
            obs = result.observation
            feedback = json.dumps(obs.last_tool_result, default=str)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool result: {feedback}\n"
                        f"Remaining steps: {obs.remaining_step_budget}\n"
                        f"Inbox: {json.dumps(obs.inbox_summary.model_dump(), default=str)}"
                    ),
                }
            )
        except Exception as err:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Error: {err}. "
                        "Respond with a valid EmailTriageAction JSON object."
                    ),
                }
            )

    return episode_id


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_baseline(
    model: str = DEFAULT_BASELINE_MODEL,
    api_key: Optional[str] = None,
    backend: str = "auto",
    session_factory: Optional[Callable[[], Any]] = None,
    grader_fetcher: Optional[Callable[[str], Any]] = None,
) -> BaselineReport:
    """Run the baseline against all 3 tasks and return aggregate scores.

    Parameters
    ----------
    model:
        LLM model identifier (used only when *backend* is ``"openai"``).
    api_key:
        OpenAI API key (used only when *backend* is ``"openai"``).
    backend:
        ``"scripted"`` — deterministic keyword-based baseline (no API key).
        ``"openai"``   — LLM baseline (requires *api_key*).
        ``"auto"``     — uses ``"scripted"`` if *api_key* is absent.
    session_factory:
        Callable returning an async context manager with ``reset()`` / ``step()``.
    grader_fetcher:
        Async callable ``(episode_id: str) -> GraderReport``.
    """
    use_scripted = backend == "scripted" or (backend == "auto" and not api_key)

    task_results: List[BaselineTaskResult] = []

    for task_id in TASK_ORDER:
        async with session_factory() as session:
            if use_scripted:
                episode_id = await _run_scripted_episode(session, task_id)
            else:
                episode_id = await _run_llm_episode(session, task_id, model, api_key)

        report: GraderReport = await grader_fetcher(episode_id)
        task_results.append(
            BaselineTaskResult(
                task_id=task_id,
                episode_id=episode_id,
                score=report.score,
                cumulative_reward=report.cumulative_reward,
                step_count=report.step_count,
            )
        )

    overall = (
        sum(r.score for r in task_results) / len(task_results)
        if task_results
        else 0.0
    )

    return BaselineReport(
        model="scripted-baseline" if use_scripted else model,
        overall_score=round(overall, 4),
        tasks=task_results,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point — prints baseline scores to stdout."""
    parser = argparse.ArgumentParser(
        description="Run the EmailTriage scripted baseline."
    )
    parser.add_argument("--model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument(
        "--backend",
        default="scripted",
        choices=["auto", "scripted", "openai"],
    )
    args = parser.parse_args()

    print(
        "Use the /baseline endpoint of a running server to execute the baseline.\n"
        f"  Example: uvicorn server.app:app --port 8000\n"
        f"  Then:    curl 'http://localhost:8000/baseline?backend={args.backend}'\n"
    )
