---
title: EmailTriage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
short_description: RL benchmark for AI agents triaging emails
tags:
  - openenv
  - agent-environment
  - reinforcement-learning
  - email-triage
  - customer-support
  - benchmark
license: mit
---

# EmailTriage — OpenEnv Environment

Ever wondered how good an AI agent actually is at handling a messy inbox? EmailTriage is a benchmark that throws realistic customer-support emails at your agent and scores how well it routes, prioritizes, tags, and responds to them — all under a step budget. Think of it as a gym for email-handling agents, built on the [OpenEnv spec](https://github.com/meta-pytorch/OpenEnv).

It's **deterministic** (same emails, same ground truth every run), **multi-objective** (speed vs. accuracy vs. response quality), and ships as a ready-to-use Docker-based [Hugging Face Space](https://huggingface.co/spaces/isodiscrete/email-triage-openenv).

---

## Environment Description

Customer support teams process hundreds of emails daily.  Doing this well requires:
- **Routing accuracy** — sending each email to the right department
- **Priority judgment** — identifying what is critical vs. low-urgency
- **Spam filtering** — discarding junk without blocking legitimate mail
- **Response quality** — drafting professional, empathetic replies that address the customer's concerns

These objectives interact and trade off: a fast agent may route incorrectly; a thorough agent may exceed the step budget.  This multi-objective structure makes EmailTriage a useful benchmark for RLHF and multi-step reasoning research.

---

## Action Space

Actions are JSON objects conforming to `EmailTriageAction`:

| Tool | Required Fields | Description |
|------|----------------|-------------|
| `read_email` | `email_id` | Reveal full email content (subject, body, metadata) |
| `classify_email` | `email_id`, `department` | Route email to a department |
| `set_priority` | `email_id`, `priority` | Assign priority level |
| `add_tags` | `email_id`, `tags` | Apply semantic tags |
| `draft_response` | `email_id`, `response_text` | Submit a customer-facing response draft |
| `mark_spam` | `email_id` | Flag email as spam (no response needed) |
| `escalate_email` | `email_id`, `escalation_reason` | Escalate with a reason |

**Department labels:** `Billing` · `Technical Support` · `Account Management` · `Sales` · `HR` · `General`

**Priority labels:** `Critical` · `High` · `Medium` · `Low`

**Tag labels:** `complaint` · `inquiry` · `refund` · `password-reset` · `order-status` · `technical-issue` · `account-closure` · `payment-failed` · `spam` · `feedback` · `billing-dispute` · `service-outage` · `feature-request`

---

## Observation Space

Each `step()` returns an `EmailTriageObservation` containing:

| Field | Description |
|-------|-------------|
| `episode_id` | Unique episode identifier |
| `task_id` / `task_title` / `task_objective` | Current task metadata |
| `inbox_summary` | Aggregated inbox state (counts + per-email summaries) |
| `active_email` | Full email content (populated after `read_email`) |
| `last_tool_result` | Structured output of the most recent tool call |
| `remaining_step_budget` | Steps left before the episode ends |
| `score_summary` | Running score estimate and processing counts |
| `reward_breakdown` | Step reward with named bonuses and penalties |
| `reward` | Scalar step reward |
| `done` | Whether the episode has ended |

---

## Tasks

### Task 1: `routing_easy` (easy)
**Objective:** Classify 5 emails to the correct department.

- 5 emails covering Billing, Technical Support, Account Management, Sales, and HR
- Step budget: 20
- Allowed tools: `read_email`, `classify_email`, `mark_spam`, `escalate_email`
- Score: routing accuracy (fraction of correct departments)

### Task 2: `priority_medium` (medium)
**Objective:** Assign the correct priority to 6 emails and detect 1 spam email.

- 6 emails including 2 Critical, 1 spam, and varied priorities
- Step budget: 35
- Allowed tools: all except `draft_response`
- Score: `0.5 × priority_accuracy + 0.3 × spam_accuracy + 0.2 × tag_F1`

### Task 3: `drafting_hard` (hard)
**Objective:** Perform full triage on 4 emails: route, prioritize, tag, and draft a professional response.

- 4 emails with complex, high-stakes scenarios (billing dispute, API failure, account closure, VIP payment failure)
- Step budget: 40
- Allowed tools: all 7
- Score: `0.25 × routing + 0.25 × priority + 0.10 × tag_F1 + 0.40 × response_quality`

---

## Reward Function

Rewards are provided at every step (not just at episode end), giving continuous signal over the trajectory:

| Action | Reward / Penalty |
|--------|-----------------|
| `read_email` | +0.01 (encourages reading before acting) |
| Correct department | **+0.20** |
| Wrong department | −0.05 |
| Correct priority (exact) | **+0.15** |
| Adjacent priority tier | +0.05 |
| Priority 2+ tiers off | −0.05 × distance |
| SLA timing bonus (Critical/High resolved early) | +0.05 |
| Correct tag | +0.05 per tag |
| Wrong tag | −0.02 per tag |
| Response quality | **+0.40 × keyword_coverage** |
| Professional tone (greeting + closing) | +0.05 |
| Unprofessional tone | −0.05 |
| True spam detected | **+0.10** |
| False spam positive | −0.10 |
| Correct escalation | +0.05 |
| Unnecessary escalation | −0.03 |
| Duplicate action | −0.02 |
| Invalid email ID | −0.02 |

---

## Baseline Scores

The scripted keyword-based baseline (no API key required) achieves:

| Task | Score |
|------|-------|
| `routing_easy` | ~0.60 |
| `priority_medium` | ~0.50 |
| `drafting_hard` | ~0.38 |
| **Overall** | **~0.49** |

Run it yourself:
```bash
curl 'http://localhost:8000/baseline?backend=scripted'
```

---

## Setup & Usage

### Prerequisites
- Python ≥ 3.10
- `pip install 'openenv-core[core]>=0.2.2'`

### Install

```bash
git clone https://huggingface.co/spaces/<your-username>/email-triage-env
cd email-triage-env
pip install -e ".[dev]"
```

### Run the server

```bash
uvicorn server.app:app --port 8000
```

### Interact via HTTP

```bash
# List tasks
curl http://localhost:8000/tasks

# Reset an episode
curl -X POST http://localhost:8000/reset \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "routing_easy"}'

# Take a step
curl -X POST http://localhost:8000/step \
  -H 'Content-Type: application/json' \
  -d '{"episode_id": "<id>", "action": {"tool": "read_email", "email_id": "e001"}}'

# Get grader report
curl 'http://localhost:8000/grader?episode_id=<id>'

# Run scripted baseline (no API key)
curl 'http://localhost:8000/baseline?backend=scripted'

# Run LLM baseline (requires OPENAI_API_KEY)
OPENAI_API_KEY=sk-... curl 'http://localhost:8000/baseline?backend=openai&model=gpt-4o-mini'
```

### Use the Python client

```python
import asyncio
from client import EmailTriageEnv
from models import EmailTriageAction

async def main():
    async with EmailTriageEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task_id="routing_easy")
        obs = result.observation
        print(f"Episode: {obs.episode_id}")
        print(f"Inbox: {obs.inbox_summary.total_emails} emails")

        result = await env.step(
            EmailTriageAction(tool="read_email", email_id="e001")
        )
        print(result.observation.active_email.subject)

asyncio.run(main())
```

### TRL / GRPO training integration

```python
from trl import GRPOTrainer
from server.email_triage_environment import EmailTriageEnvironment
from email_core import EmailTaskCatalog, EpisodeRegistry

catalog = EmailTaskCatalog()
registry = EpisodeRegistry()

trainer = GRPOTrainer(
    model=model,
    environment_factory=lambda: EmailTriageEnvironment(catalog, registry),
    reward_funcs=[lambda environments, **kw: [e._runtime.cumulative_reward for e in environments]],
    ...
)
```

### Run tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=term-missing
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 8000:8000 email-triage-env

# With LLM baseline support
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... email-triage-env
```

### Validate with openenv CLI

```bash
openenv validate
```

---

## Repository Structure

```
email-triage-env/
├── __init__.py                        # Package re-exports
├── models.py                          # Pydantic types (Action / Observation / State)
├── client.py                          # Typed async EnvClient subclass
├── email_core.py                      # Deterministic engine (rewards, grader, registry)
├── baseline.py                        # Scripted + LLM baseline runners
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Package config
├── Dockerfile                         # Multi-stage container build
├── data/
│   └── emails.json                    # 20 labelled emails + task definitions
├── server/
│   ├── email_triage_environment.py    # Environment class (reset / step / state)
│   └── app.py                         # FastAPI app (/tasks /grader /baseline)
└── tests/
    ├── test_models.py                 # Pydantic model validation tests
    ├── test_core.py                   # Engine unit tests (60+ assertions)
    └── test_app.py                    # API endpoint smoke tests
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `EMAILTRIAGE_HOST` | `0.0.0.0` | Server bind address |
| `EMAILTRIAGE_MAX_CONCURRENT_ENVS` | `4` | Max parallel sessions |
| `EMAILTRIAGE_BASELINE_BACKEND` | `auto` | Default baseline backend |
| `OPENAI_API_KEY` | — | Required for LLM baseline |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model for baseline |
| `OPENAI_BASE_URL` | — | Custom OpenAI-compatible endpoint |

---

## License

MIT
