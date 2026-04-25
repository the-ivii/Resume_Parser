# AI Resume Reviewer & Career Coach

A multi-agent AI system built with **LangChain** and **LangGraph** that reviews
a resume against a target job role and returns a polished, actionable review.

---

## Why this use case?

Resume review is a perfect fit for a multi-agent workflow:

- It decomposes naturally into **parsing**, **analysis**, **scoring**,
  **coaching**, and **writing** — each of which is a different kind of
  reasoning problem.
- The output is immediately **useful** (one of the rubric criteria).
- Every agent has a **clear, non-overlapping role**, which makes the graph
  easy to explain in the demo video.

---

## Architecture

Five specialized agents collaborate over a shared `ResumeState`:

| # | Agent | Role |
|---|---|---|
| 1 | **Parser Agent** | Extracts structured data (contact, education, experience, skills, projects) from raw resume text. |
| 2 | **Skills Analyst** | Identifies strengths, weaknesses, missing skills, and ATS keyword gaps vs. the target role. |
| 3 | **Job Fit Agent** | Produces a `match_score` (0–100), verdict, reasoning, and red flags. |
| 4 | **Improvement Advisor** | Generates 6–10 prioritized, concrete improvements (with rewritten bullets). |
| 5 | **Report Compiler** | Synthesizes everything into a polished markdown report, including a 4-week action plan. |

### LangGraph workflow

```
                ┌──────────┐
   START ─────▶ │  parser  │
                └────┬─────┘
                     ▼
             ┌───────────────┐
             │ skills_analyst│
             └───────┬───────┘
                     │  conditional edge
       target_role?  ├─────────────────┐
                 yes │                 │ no
                     ▼                 │
               ┌──────────┐            │
               │ job_fit  │            │
               └────┬─────┘            │
                    │                  │
                    ▼                  ▼
                ┌──────────────────────────┐
                │  improvement_advisor     │
                └────────────┬─────────────┘
                             ▼
                    ┌───────────────────┐
                    │  report_compiler  │
                    └────────┬──────────┘
                             ▼
                            END
```

The **conditional edge** after `skills_analyst` skips `job_fit` if the user
did not provide a target role, demonstrating real LangGraph branching.

### Shared state (`ResumeState`)

```python
class ResumeState(TypedDict, total=False):
    resume_text: str
    target_role: str
    parsed_resume: dict
    skills_analysis: dict
    job_fit: dict
    improvements: list[dict]
    final_report: str
    trace: list[str]
```

Every agent reads the fields it needs and writes only its own fields, so
state passing is explicit and auditable — LangGraph merges the partial
updates into the next node's input automatically.

---

## Setup

Requires Python 3.9+ and an Gemini API key.

```bash
# 1. Clone and enter the project
cd "untitled folder 2"   # (or the repo root)

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env
# then open .env and set OPENAI_API_KEY=sk-...
```

---

## Running

```bash
# Real run (requires OPENAI_API_KEY)
python multi_agent_system.py

# Offline demo - canned responses, no API key required
python multi_agent_system.py --mock

# Fully non-interactive (great for screencast recording)
python multi_agent_system.py \
    --mock \
    --non-interactive \
    --role "Senior Backend Engineer at a fintech startup"
```

The program will:

1. Ask you to provide a resume (sample / paste / file path).
2. Ask for a target job role (optional).
3. Run all five agents through the LangGraph workflow.
4. Print each agent's intermediate output live in the terminal.
5. Render the final markdown report.
6. Save the report to `resume_review_output.md`.

### CLI flags

| Flag | Purpose |
|---|---|
| `--mock` | Use canned responses instead of calling OpenAI. Useful for offline testing, CI, and verifying the pipeline without spending tokens. |
| `--resume <path>` | Provide a resume file path non-interactively. |
| `--role "<title>"` | Provide the target role non-interactively. |
| `--non-interactive` | Skip all prompts. Combines well with `--resume` / `--role`. |

### Example session

```text
$ python multi_agent_system.py

  AI Resume Reviewer & Career Coach
  Multi-Agent System (LangChain + LangGraph)

── Step 1: Provide the resume ──
Choose input method:
  1) Use the bundled sample resume
  2) Paste resume text
  3) Load resume from a file path
Your choice [1/2/3] (default 1): 1

── Step 2: Target role ──
Enter the target job role: Senior Backend Engineer at a fintech startup

── Running the agent graph ──
── Agent 1: Parser ──          ✓ extracted 2 experience entries, 10 skills
── Agent 2: Skills Analyst ──  ✓ 5 strengths, 4 missing skills
── Agent 3: Job Fit ──         Match score: 62/100   Verdict: Partial Match
── Agent 4: Improvement Advisor ── ✓ 8 improvements
── Agent 5: Report Compiler ── ✓ final markdown ready

# Resume Review Report
...
```

---

## Files

| File | Purpose |
|---|---|
| `multi_agent_system.py` | **Main file required by the assignment.** All agents + graph. |
| `requirements.txt` | Pinned Python dependencies. |
| `.env.example` | Template for your OpenAI key. |
| `sample_resume.txt` | Ready-to-use resume so the demo works out of the box. |
| `README.md` | This file. |
| `DEMO_SCRIPT.md` | Voice-over script for the 5–8 minute demo video. |
| `resume_review_output.md` | Generated on each run — the final compiled report. |

---

## Key learnings

- **LangGraph's `StateGraph` + `TypedDict`** make multi-agent coordination
  easier than hand-rolling a dispatcher — each node is just a pure function.
- **Conditional edges** are how you express "skip this agent when the input
  is missing" without `if` sprinkled through the agents themselves.
- **JSON-shaped prompts** make agent output composable: agent *N*'s JSON
  lands directly in agent *N+1*'s context, which is the whole point of the
  shared-state pattern.
- Keeping each agent's prompt **narrow and opinionated** dramatically
  improves the final report's quality compared to a single mega-prompt.
