"""
multi_agent_system.py
=====================

A multi-agent AI Resume Reviewer & Career Coach built with LangChain + LangGraph.

Five specialized agents collaborate over a shared state to produce a polished,
actionable resume review against a target job role.

Agents
------
1. Parser Agent             -> Extracts structured data from raw resume text.
2. Skills Analyst Agent     -> Identifies strengths, weaknesses, missing skills.
3. Job Fit Agent            -> Scores resume <-> target role match (0-100) with reasoning.
4. Improvement Advisor      -> Produces prioritized, actionable suggestions.
5. Report Compiler Agent    -> Synthesizes the final markdown report.

LangGraph Workflow
------------------
    START
      |
      v
    parser ---> skills_analyst ---> (conditional)
                                      |-- target role given --> job_fit --> advisor --> compiler --> END
                                      |-- no target role ------------------> advisor --> compiler --> END

Usage
-----
    python multi_agent_system.py

You will be prompted to paste/load a resume and enter a target job role.

Environment
-----------
Requires an OpenAI API key in the environment (or in a .env file):
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini   # optional, defaults to gpt-4o-mini
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore")

import argparse
import json
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_openai import ChatOpenAI
try:
    from langchain_groq import ChatGroq  # optional, free provider
except Exception:  # pragma: no cover
    ChatGroq = None  # type: ignore[assignment]
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # optional, free provider
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore[assignment]
from langgraph.graph import StateGraph, START, END

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.rule import Rule
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()
console = Console()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Provider selection:
# - If LLM_PROVIDER env is set ("openai" | "groq" | "gemini"), use it.
# - Else if GOOGLE_API_KEY is set, prefer Gemini (most generous free tier).
# - Else if GROQ_API_KEY is set, use Groq (also free).
# - Else fall back to OpenAI.
def _detect_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in ("openai", "groq", "gemini"):
        return explicit
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return "openai"

# --- Mock mode ---------------------------------------------------------------
# Set via CLI flag (--mock) or env var DEMO_MOCK=1. Uses canned, deterministic
# JSON responses so the graph can be verified end-to-end without calling a real
# LLM. Invaluable for tests, offline demos, and CI.
MOCK_MODE = False


def _enable_mock_mode() -> None:
    """Flip the global MOCK_MODE flag."""
    global MOCK_MODE
    MOCK_MODE = True


MOCK_RESPONSES: Dict[str, str] = {
    "parser": json.dumps(
        {
            "name": "Priya Sharma",
            "contact": {
                "email": "priya.sharma@example.com",
                "phone": "+91-98765-43210",
                "location": "Bengaluru, India",
                "links": [
                    "linkedin.com/in/priyasharma-eg",
                    "github.com/priya-s",
                ],
            },
            "summary": "Software engineer with 3 years of experience building web applications and internal tools; interested in backend-heavy roles.",
            "education": [
                {
                    "degree": "B.E. Computer Science",
                    "institution": "PES University",
                    "year": "2022",
                    "details": "CGPA 8.4/10; coursework in OS, DBMS, Distributed Systems",
                }
            ],
            "experience": [
                {
                    "title": "Software Engineer",
                    "company": "Acme Retail Tech",
                    "duration": "Jul 2022 - Present",
                    "highlights": [
                        "Built CRUD APIs in Django for the merchandising team.",
                        "Helped migrate a legacy MySQL database to PostgreSQL.",
                        "Wrote unit tests for the checkout service.",
                        "Participated in on-call rotation.",
                    ],
                },
                {
                    "title": "Software Engineering Intern",
                    "company": "BrightLabs",
                    "duration": "Jan 2022 - Jun 2022",
                    "highlights": [
                        "Worked on a React dashboard for internal analytics.",
                        "Fixed bugs in the Node.js backend.",
                    ],
                },
            ],
            "skills": {
                "technical": [
                    "Python",
                    "Django",
                    "Flask",
                    "JavaScript",
                    "React",
                    "SQL",
                    "REST",
                ],
                "soft": [],
                "tools": ["Git", "Docker", "Linux"],
            },
            "projects": [
                {
                    "name": "ExpenseSplit",
                    "description": "Web app to split bills among roommates, ~200 GitHub stars.",
                    "tech": ["Django", "React", "Postgres", "Render"],
                },
                {
                    "name": "Kaggle Titanic Challenge",
                    "description": "Feature engineering and classifier experiments; top 40% finish.",
                    "tech": ["Python", "scikit-learn"],
                },
            ],
            "certifications": ["AWS Cloud Practitioner (2023)"],
        }
    ),
    "skills": json.dumps(
        {
            "strengths": [
                "3 years of production Python + Django experience.",
                "Hands-on database migration experience (MySQL -> Postgres).",
                "Ships side projects (ExpenseSplit) with real users.",
                "Comfortable on-call, signaling production ownership.",
                "AWS Cloud Practitioner certification.",
            ],
            "weaknesses": [
                "No evidence of system-design or architecture ownership.",
                "Limited mention of scale, performance, or reliability metrics.",
                "Few quantified impact bullets on experience entries.",
                "Fintech-specific experience (payments, compliance) is absent.",
            ],
            "missing_skills": [
                "Distributed systems / messaging (Kafka, SQS)",
                "Observability (Prometheus, Grafana, tracing)",
                "Payments / ledger / fintech domain knowledge",
                "Kubernetes or advanced container orchestration",
            ],
            "keyword_gaps": [
                "microservices",
                "SLA",
                "latency",
                "idempotency",
                "PCI",
                "gRPC",
            ],
            "overall_notes": "Solid mid-level backend engineer profile. The resume reads as a strong generalist, but needs sharper quantified wins and fintech-relevant signals to land a senior backend role at a fintech startup.",
        }
    ),
    "fit": json.dumps(
        {
            "match_score": 62,
            "verdict": "Partial Match",
            "reasoning": [
                "Strong Python + Django foundation aligns with typical fintech backend stacks.",
                "Database migration experience demonstrates comfort with persistent systems.",
                "Lack of distributed-systems / messaging work is a gap for 'senior' framing.",
                "No fintech-domain keywords (payments, ledgers, compliance) on the resume.",
                "Side-project maintenance hints at ownership and shipping velocity.",
            ],
            "red_flags": [
                "Senior title requested but bullets read mid-level; few metrics.",
                "No evidence of mentoring or leading technical initiatives.",
            ],
            "quick_wins": [
                "Quantify impact on every experience bullet (latency, throughput, $ saved).",
                "Add a 'Systems' subsection listing any queue / cache / observability work.",
                "Reframe the Postgres migration with scale and downtime numbers.",
            ],
        }
    ),
    "advisor": json.dumps(
        [
            {
                "priority": "High",
                "category": "Experience bullets",
                "issue": "Bullets describe activities, not outcomes. Example: 'Built CRUD APIs in Django'.",
                "suggestion": "Rewrite with action + scope + metric. Example: 'Built 14 Django REST endpoints powering the merchandising console used by 120+ internal users, reducing manual ops work by ~30%.'",
            },
            {
                "priority": "High",
                "category": "Keywords/ATS",
                "issue": "Missing fintech- and senior-backend-relevant keywords.",
                "suggestion": "Weave in: microservices, idempotency, SLA, latency, observability, CI/CD, gRPC, Kafka/SQS where honestly applicable.",
            },
            {
                "priority": "High",
                "category": "Summary",
                "issue": "Summary is generic and doesn't position for a senior fintech role.",
                "suggestion": "Rewrite: 'Backend engineer with 3 years shipping Django services and owning a production Postgres migration. Targeting senior backend roles at fintechs building reliable payments infrastructure.'",
            },
            {
                "priority": "High",
                "category": "Experience bullets",
                "issue": "MySQL -> Postgres migration is the strongest story but is one sterile line.",
                "suggestion": "Expand to 2-3 bullets: why (OLTP/perf), scope (DB size, tables, downtime budget), outcome (query latency drop, zero-downtime cutover).",
            },
            {
                "priority": "Medium",
                "category": "Skills",
                "issue": "Skills list is flat and mixes levels.",
                "suggestion": "Split into 'Languages', 'Frameworks', 'Infra', 'Tools' and drop anything still labeled '(basics)' - it flags weakness.",
            },
            {
                "priority": "Medium",
                "category": "Projects",
                "issue": "ExpenseSplit project is underleveraged.",
                "suggestion": "Add bullets: users (~X weekly active), infra (Render, Postgres), one interesting technical decision (e.g. how splits are reconciled).",
            },
            {
                "priority": "Medium",
                "category": "Structure",
                "issue": "No dedicated 'Impact' or 'Systems' section.",
                "suggestion": "Create a 3-line 'Selected Impact' block at the top summarizing the migration, the side project, and any on-call wins.",
            },
            {
                "priority": "Low",
                "category": "Other",
                "issue": "Kaggle Titanic project is dated and not role-relevant.",
                "suggestion": "Either drop it or replace with a more recent backend-flavored project.",
            },
        ]
    ),
    "compiler": textwrap.dedent(
        """
        # Resume Review Report

        **Candidate:** Priya Sharma
        **Target role:** Senior Backend Engineer at a fintech startup
        **Generated:** MOCK RUN

        ## 1. Candidate Snapshot
        Priya is a mid-level software engineer with three years of production
        Python and Django experience at Acme Retail Tech, complemented by a
        strong open-source side project (ExpenseSplit, ~200 GitHub stars) and
        an AWS Cloud Practitioner certification. Her strongest signal is a
        successful MySQL-to-Postgres migration, but the resume currently
        under-sells this work.

        ## 2. Match Verdict
        - **Score:** 62/100
        - **Verdict:** Partial Match
        - Strong Python/Django foundation aligns with fintech stacks.
        - Missing distributed-systems and messaging experience.
        - No fintech-domain signals (payments, ledgers, compliance).
        - Shipping side projects demonstrates ownership.

        ## 3. Strengths
        - Three years of production Python + Django.
        - Real database migration experience (MySQL -> Postgres).
        - Ships side projects with real users.
        - Comfortable being on-call in production.
        - AWS Cloud Practitioner certified.

        ## 4. Weaknesses & Gaps
        - Few quantified impact metrics on experience bullets.
        - No system-design or architecture ownership signals.
        - No fintech-domain experience or keywords.
        - Missing: distributed systems, observability, Kubernetes.
        - ATS gaps: `microservices`, `SLA`, `idempotency`, `latency`, `PCI`.

        ## 5. Prioritized Improvements
        1. **[High] Experience bullets:** Rewrite bullets with action + scope
           + metric. e.g. *"Built 14 Django REST endpoints powering the
           merchandising console used by 120+ internal users, reducing manual
           ops work by ~30%."*
        2. **[High] Keywords/ATS:** Add fintech- and senior-relevant terms
           where honestly applicable: microservices, idempotency, SLA,
           latency, observability, CI/CD, gRPC, Kafka/SQS.
        3. **[High] Summary:** Rewrite to position for seniority and fintech:
           *"Backend engineer with 3 years shipping Django services and
           owning a production Postgres migration. Targeting senior backend
           roles at fintechs building reliable payments infrastructure."*
        4. **[High] Experience bullets:** Expand the MySQL -> Postgres
           migration into 2-3 bullets covering why, scope (DB size, tables,
           downtime budget), and outcome (latency drop, zero-downtime).
        5. **[Medium] Skills:** Split flat list into Languages / Frameworks /
           Infra / Tools and drop any `(basics)` labels.
        6. **[Medium] Projects:** Flesh out ExpenseSplit with WAU, infra,
           and one interesting technical decision.
        7. **[Medium] Structure:** Add a 3-line *Selected Impact* block at
           the top.
        8. **[Low] Other:** Drop or replace the dated Kaggle Titanic entry.

        ## 6. 30-Day Action Plan
        **Week 1 - Reframe**
        - Rewrite summary for senior fintech positioning.
        - Add *Selected Impact* block at the top.

        **Week 2 - Quantify**
        - Rewrite every experience bullet with action + scope + metric.
        - Expand the Postgres migration into a 3-bullet story.

        **Week 3 - Close keyword gaps**
        - Audit against fintech JDs and weave in missing keywords honestly.
        - Split the skills section into categories.

        **Week 4 - Projects & polish**
        - Upgrade ExpenseSplit write-up; drop or refresh Titanic.
        - Have 2 peers do a 10-minute read-through and iterate.

        ## 7. Final Encouragement
        You already have the raw material of a strong senior backend
        candidate - a real migration story, a shipped side project, and
        production on-call experience. With one focused week of quantifying
        and repositioning, this resume can cross the line from "interesting
        generalist" to "obvious fintech backend hire." You've got this.
        """
    ).strip(),
}


def build_llm(temperature: float = 0.2, agent_key: str = "parser"):
    """Build an LLM for an agent.

    - In mock mode: returns a `FakeListChatModel` with canned JSON responses.
    - If GROQ_API_KEY is set (or LLM_PROVIDER=groq): returns `ChatGroq`
      (free tier, great for this assignment).
    - Otherwise: returns `ChatOpenAI`.
    """
    if MOCK_MODE:
        return FakeListChatModel(responses=[MOCK_RESPONSES[agent_key]])

    provider = _detect_provider()

    if provider == "gemini":
        if ChatGoogleGenerativeAI is None:
            console.print("[red]langchain-google-genai is not installed. Run: pip install langchain-google-genai[/red]")
            sys.exit(1)
        if not os.getenv("GOOGLE_API_KEY"):
            console.print(
                Panel(
                    "[bold red]GOOGLE_API_KEY is not set.[/bold red]\n\n"
                    "Get a free key at [cyan]https://aistudio.google.com/apikey[/cyan]\n"
                    "then add to your [cyan].env[/cyan]:\n"
                    "  [green]GOOGLE_API_KEY=AIza...[/green]",
                    title="Missing Google key",
                    border_style="red",
                )
            )
            sys.exit(1)
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)

    if provider == "groq":
        if ChatGroq is None:
            console.print("[red]langchain-groq is not installed. Run: pip install langchain-groq[/red]")
            sys.exit(1)
        if not os.getenv("GROQ_API_KEY"):
            console.print(
                Panel(
                    "[bold red]GROQ_API_KEY is not set.[/bold red]\n\n"
                    "Get a free key at [cyan]https://console.groq.com/keys[/cyan]\n"
                    "then add to your [cyan].env[/cyan]:\n"
                    "  [green]GROQ_API_KEY=gsk_...[/green]",
                    title="Missing Groq key",
                    border_style="red",
                )
            )
            sys.exit(1)
        return ChatGroq(model=GROQ_MODEL, temperature=temperature)

    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            Panel(
                "[bold red]OPENAI_API_KEY is not set.[/bold red]\n\n"
                "Create a [cyan].env[/cyan] file with:\n"
                "  [green]OPENAI_API_KEY=sk-...[/green]\n\n"
                "Tips:\n"
                "  - Use Groq (free) by setting [green]GROQ_API_KEY=gsk_...[/green] instead.\n"
                "  - Run offline with [cyan]python multi_agent_system.py --mock[/cyan].",
                title="Missing API key",
                border_style="red",
            )
        )
        sys.exit(1)
    return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature)


def _parse_json_safely(raw: str) -> Any:
    """Pull the first JSON object/array out of an LLM response, tolerating fences."""
    txt = raw.strip()
    # Strip ```json ... ``` or ``` ... ``` fences if present
    if txt.startswith("```"):
        txt = txt.strip("`")
        if txt.lower().startswith("json"):
            txt = txt[4:]
        txt = txt.strip()
        if txt.endswith("```"):
            txt = txt[:-3].strip()
    # Find first { or [ and last matching }/]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = txt.find(open_c)
        end = txt.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(txt[start : end + 1])
            except json.JSONDecodeError:
                pass
    # Last-ditch effort
    return json.loads(txt)


# ---------------------------------------------------------------------------
# Shared State
# ---------------------------------------------------------------------------


class ResumeState(TypedDict, total=False):
    """State passed between every agent in the graph."""

    # Inputs
    resume_text: str
    target_role: str

    # Produced by agents
    parsed_resume: Dict[str, Any]
    skills_analysis: Dict[str, Any]
    job_fit: Dict[str, Any]
    improvements: List[Dict[str, Any]]
    final_report: str

    # Trace of work for observability / the video demo
    trace: List[str]


# ---------------------------------------------------------------------------
# Agent 1: Parser
# ---------------------------------------------------------------------------

PARSER_SYSTEM = """You are a precise resume parsing agent.
Extract structured data from the resume text the user provides.

Return ONLY valid JSON with this exact schema:
{
  "name": "string or null",
  "contact": {"email": "string or null", "phone": "string or null", "location": "string or null", "links": ["string", ...]},
  "summary": "string or null",
  "education": [{"degree": "string", "institution": "string", "year": "string", "details": "string or null"}],
  "experience": [{"title": "string", "company": "string", "duration": "string", "highlights": ["string", ...]}],
  "skills": {"technical": ["string", ...], "soft": ["string", ...], "tools": ["string", ...]},
  "projects": [{"name": "string", "description": "string", "tech": ["string", ...]}],
  "certifications": ["string", ...]
}
If a section is missing, use null or an empty list. Do not invent data.
"""


def parser_agent(state: ResumeState) -> ResumeState:
    """Agent 1 — converts raw resume text into a structured dictionary."""
    console.print(Rule("[bold cyan]Agent 1: Parser[/bold cyan]"))
    llm = build_llm(temperature=0.0, agent_key="parser")
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=PARSER_SYSTEM),
            ("human", "Resume text:\n\n{resume_text}"),
        ]
    )
    chain = prompt | llm
    with Progress(SpinnerColumn(), TextColumn("[cyan]Parsing resume..."), transient=True) as p:
        p.add_task("parse", total=None)
        response = chain.invoke({"resume_text": state["resume_text"]})

    try:
        parsed = _parse_json_safely(response.content)
    except Exception as e:  # pragma: no cover
        console.print(f"[yellow]Parser fallback: {e}[/yellow]")
        parsed = {"raw": response.content}

    trace = state.get("trace", []) + [
        f"Parser extracted {len(parsed.get('experience', []) or [])} experience entries, "
        f"{len((parsed.get('skills', {}) or {}).get('technical', []) or [])} technical skills."
    ]
    console.print(
        Panel.fit(
            f"[bold]Name:[/bold] {parsed.get('name')}\n"
            f"[bold]Email:[/bold] {(parsed.get('contact') or {}).get('email')}\n"
            f"[bold]Experience entries:[/bold] {len(parsed.get('experience', []) or [])}\n"
            f"[bold]Skills found:[/bold] {len((parsed.get('skills', {}) or {}).get('technical', []) or [])} technical",
            title="Parser Output",
            border_style="cyan",
        )
    )
    return {"parsed_resume": parsed, "trace": trace}


# ---------------------------------------------------------------------------
# Agent 2: Skills Analyst
# ---------------------------------------------------------------------------

SKILLS_SYSTEM = """You are a senior technical recruiter acting as a skills analyst.
Given a parsed resume (JSON) and an optional target role, produce a rigorous
strengths / weaknesses / missing-skills analysis.

Return ONLY valid JSON:
{
  "strengths": ["concise bullet", ...],       // 3-6 items
  "weaknesses": ["concise bullet", ...],      // 2-5 items
  "missing_skills": ["skill", ...],           // skills relevant for the target role that are absent
  "keyword_gaps": ["keyword", ...],           // ATS-relevant keywords missing
  "overall_notes": "one paragraph"
}
Be specific and grounded in the resume data. Do not invent work experience.
"""


def skills_analyst_agent(state: ResumeState) -> ResumeState:
    """Agent 2 — reasons about skill strengths, weaknesses, and gaps."""
    console.print(Rule("[bold magenta]Agent 2: Skills Analyst[/bold magenta]"))
    llm = build_llm(temperature=0.3, agent_key="skills")
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SKILLS_SYSTEM),
            (
                "human",
                "Target role: {target_role}\n\nParsed resume JSON:\n{parsed}",
            ),
        ]
    )
    chain = prompt | llm
    with Progress(SpinnerColumn(), TextColumn("[magenta]Analyzing skills..."), transient=True) as p:
        p.add_task("skills", total=None)
        response = chain.invoke(
            {
                "target_role": state.get("target_role") or "No specific role provided",
                "parsed": json.dumps(state.get("parsed_resume", {}), indent=2),
            }
        )
    try:
        analysis = _parse_json_safely(response.content)
    except Exception:
        analysis = {"overall_notes": response.content}

    trace = state.get("trace", []) + [
        f"Skills analyst found {len(analysis.get('strengths', []))} strengths, "
        f"{len(analysis.get('missing_skills', []))} missing skills."
    ]

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Strengths", style="green")
    table.add_column("Weaknesses", style="yellow")
    table.add_column("Missing for role", style="red")
    max_len = max(
        len(analysis.get("strengths", [])),
        len(analysis.get("weaknesses", [])),
        len(analysis.get("missing_skills", [])),
        1,
    )
    for i in range(max_len):
        table.add_row(
            analysis.get("strengths", [""] * max_len)[i] if i < len(analysis.get("strengths", [])) else "",
            analysis.get("weaknesses", [""] * max_len)[i] if i < len(analysis.get("weaknesses", [])) else "",
            analysis.get("missing_skills", [""] * max_len)[i] if i < len(analysis.get("missing_skills", [])) else "",
        )
    console.print(table)

    return {"skills_analysis": analysis, "trace": trace}


# ---------------------------------------------------------------------------
# Agent 3: Job Fit
# ---------------------------------------------------------------------------

JOB_FIT_SYSTEM = """You are an impartial hiring manager scoring a candidate for a target role.

Return ONLY valid JSON:
{
  "match_score": integer 0-100,
  "verdict": "Strong Match" | "Good Match" | "Partial Match" | "Weak Match",
  "reasoning": ["bullet", ...],   // 4-6 bullets, positives AND negatives
  "red_flags": ["bullet", ...],   // can be []
  "quick_wins": ["bullet", ...]   // 2-4 changes that would materially raise the score
}
Base your score on evidence from the parsed resume. Be honest and calibrated.
"""


def job_fit_agent(state: ResumeState) -> ResumeState:
    """Agent 3 — quantifies resume fit for the provided target role."""
    console.print(Rule("[bold yellow]Agent 3: Job Fit[/bold yellow]"))
    llm = build_llm(temperature=0.2, agent_key="fit")
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=JOB_FIT_SYSTEM),
            (
                "human",
                "Target role: {target_role}\n\n"
                "Parsed resume JSON:\n{parsed}\n\n"
                "Skills analysis JSON:\n{skills}",
            ),
        ]
    )
    chain = prompt | llm
    with Progress(SpinnerColumn(), TextColumn("[yellow]Scoring fit..."), transient=True) as p:
        p.add_task("fit", total=None)
        response = chain.invoke(
            {
                "target_role": state["target_role"],
                "parsed": json.dumps(state.get("parsed_resume", {}), indent=2),
                "skills": json.dumps(state.get("skills_analysis", {}), indent=2),
            }
        )
    try:
        fit = _parse_json_safely(response.content)
    except Exception:
        fit = {"match_score": 0, "verdict": "Unknown", "reasoning": [response.content]}

    score = fit.get("match_score", 0)
    color = "green" if score >= 75 else "yellow" if score >= 55 else "red"
    console.print(
        Panel.fit(
            f"[bold {color}]Match score: {score}/100[/bold {color}]\n"
            f"[bold]Verdict:[/bold] {fit.get('verdict')}\n"
            f"[bold]Red flags:[/bold] {len(fit.get('red_flags', []))}",
            title="Job Fit Output",
            border_style=color,
        )
    )
    trace = state.get("trace", []) + [
        f"Job fit agent scored {score}/100 ({fit.get('verdict')})."
    ]
    return {"job_fit": fit, "trace": trace}


# ---------------------------------------------------------------------------
# Agent 4: Improvement Advisor
# ---------------------------------------------------------------------------

ADVISOR_SYSTEM = """You are a top-tier career coach producing prioritized, concrete
resume improvements. Use the parsed resume, skills analysis, and job-fit data.

Return ONLY valid JSON - a JSON array of improvement objects:
[
  {
    "priority": "High" | "Medium" | "Low",
    "category": "Experience bullets" | "Skills" | "Structure" | "Keywords/ATS" | "Summary" | "Projects" | "Other",
    "issue": "what is wrong or missing",
    "suggestion": "concrete, actionable fix. If rewriting a bullet, provide an example rewrite."
  },
  ...
]
Return 6-10 items total, sorted High -> Low. Be specific and quote resume text when rewriting.
"""


def improvement_advisor_agent(state: ResumeState) -> ResumeState:
    """Agent 4 — generates prioritized, actionable improvements."""
    console.print(Rule("[bold green]Agent 4: Improvement Advisor[/bold green]"))
    llm = build_llm(temperature=0.4, agent_key="advisor")
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=ADVISOR_SYSTEM),
            (
                "human",
                "Target role: {target_role}\n\n"
                "Parsed resume JSON:\n{parsed}\n\n"
                "Skills analysis JSON:\n{skills}\n\n"
                "Job fit JSON:\n{fit}",
            ),
        ]
    )
    chain = prompt | llm
    with Progress(SpinnerColumn(), TextColumn("[green]Drafting improvements..."), transient=True) as p:
        p.add_task("advisor", total=None)
        response = chain.invoke(
            {
                "target_role": state.get("target_role") or "No specific role provided",
                "parsed": json.dumps(state.get("parsed_resume", {}), indent=2),
                "skills": json.dumps(state.get("skills_analysis", {}), indent=2),
                "fit": json.dumps(state.get("job_fit", {}), indent=2),
            }
        )
    try:
        improvements = _parse_json_safely(response.content)
        if not isinstance(improvements, list):
            improvements = [improvements]
    except Exception:
        improvements = [{"priority": "High", "category": "Other", "issue": "Parse error",
                         "suggestion": response.content}]

    table = Table(title="Top improvements", show_header=True, header_style="bold green")
    table.add_column("#", width=3)
    table.add_column("Priority", width=8)
    table.add_column("Category", width=18)
    table.add_column("Issue -> Suggestion")
    for i, item in enumerate(improvements[:6], 1):
        table.add_row(
            str(i),
            item.get("priority", "-"),
            item.get("category", "-"),
            f"[bold]{item.get('issue','')}[/bold]\n{item.get('suggestion','')}",
        )
    console.print(table)

    trace = state.get("trace", []) + [f"Advisor produced {len(improvements)} improvements."]
    return {"improvements": improvements, "trace": trace}


# ---------------------------------------------------------------------------
# Agent 5: Report Compiler
# ---------------------------------------------------------------------------

COMPILER_SYSTEM = """You are a professional career report writer. Given the full state
(parsed resume, skills analysis, job fit, improvements), produce a polished
markdown report for the candidate.

Use this structure exactly (filled in with real content):

# Resume Review Report
**Candidate:** <name>
**Target role:** <target role or "Not specified">
**Generated:** <today's date>

## 1. Candidate Snapshot
One short paragraph summarizing who this candidate is.

## 2. Match Verdict
- **Score:** X/100
- **Verdict:** <verdict>
- Bullet reasoning.

## 3. Strengths
Bulleted strengths.

## 4. Weaknesses & Gaps
Bulleted weaknesses, missing skills, and keyword gaps.

## 5. Prioritized Improvements
Numbered list. For each, include priority tag [High], [Medium], or [Low],
the issue, and a concrete suggestion (with rewritten example bullet when relevant).

## 6. 30-Day Action Plan
Week-by-week, 4 weeks total, 2-3 actions per week pulled from the improvements.

## 7. Final Encouragement
One short motivating paragraph.

Use clean, readable markdown. Do NOT wrap the output in code fences.
"""


def report_compiler_agent(state: ResumeState) -> ResumeState:
    """Agent 5 — produces the final polished markdown report."""
    console.print(Rule("[bold blue]Agent 5: Report Compiler[/bold blue]"))
    llm = build_llm(temperature=0.3, agent_key="compiler")
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=COMPILER_SYSTEM),
            (
                "human",
                "Today: {today}\n\n"
                "Parsed resume JSON:\n{parsed}\n\n"
                "Skills analysis JSON:\n{skills}\n\n"
                "Job fit JSON:\n{fit}\n\n"
                "Improvements JSON:\n{improvements}\n\n"
                "Target role: {target_role}",
            ),
        ]
    )
    chain = prompt | llm
    with Progress(SpinnerColumn(), TextColumn("[blue]Compiling final report..."), transient=True) as p:
        p.add_task("compile", total=None)
        response = chain.invoke(
            {
                "today": datetime.now().strftime("%B %d, %Y"),
                "parsed": json.dumps(state.get("parsed_resume", {}), indent=2),
                "skills": json.dumps(state.get("skills_analysis", {}), indent=2),
                "fit": json.dumps(state.get("job_fit", {}), indent=2),
                "improvements": json.dumps(state.get("improvements", []), indent=2),
                "target_role": state.get("target_role") or "Not specified",
            }
        )

    report = response.content.strip()
    # Strip accidental code fences
    if report.startswith("```"):
        report = report.strip("`")
        if report.lower().startswith("markdown"):
            report = report[len("markdown"):].lstrip()

    trace = state.get("trace", []) + [f"Compiler produced {len(report)} chars of markdown."]
    return {"final_report": report, "trace": trace}


# ---------------------------------------------------------------------------
# Graph assembly (LangGraph)
# ---------------------------------------------------------------------------


def route_after_skills(state: ResumeState) -> str:
    """Conditional edge: skip the fit scorer if no target role was provided."""
    if (state.get("target_role") or "").strip():
        return "fit_node"
    return "advisor_node"


def build_graph():
    """Build and compile the LangGraph multi-agent workflow."""
    graph = StateGraph(ResumeState)

    graph.add_node("parser_node", parser_agent)
    graph.add_node("skills_node", skills_analyst_agent)
    graph.add_node("fit_node", job_fit_agent)
    graph.add_node("advisor_node", improvement_advisor_agent)
    graph.add_node("compiler_node", report_compiler_agent)

    graph.add_edge(START, "parser_node")
    graph.add_edge("parser_node", "skills_node")
    graph.add_conditional_edges(
        "skills_node",
        route_after_skills,
        {"fit_node": "fit_node", "advisor_node": "advisor_node"},
    )
    graph.add_edge("fit_node", "advisor_node")
    graph.add_edge("advisor_node", "compiler_node")
    graph.add_edge("compiler_node", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# CLI / user input
# ---------------------------------------------------------------------------


SAMPLE_RESUME_PATH = Path(__file__).parent / "sample_resume.txt"


def _read_multiline(prompt: str) -> str:
    """Read multi-line input from stdin until a line with only 'END'."""
    console.print(
        f"[cyan]{prompt}[/cyan]\n"
        "[dim](paste your text, then type END on a new line and press Enter)[/dim]"
    )
    lines: List[str] = []
    try:
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines).strip()


def get_resume_text() -> str:
    """Ask the user how they want to provide the resume, and return the text."""
    console.print(Rule("[bold]Step 1: Provide the resume[/bold]"))
    console.print(
        "Choose input method:\n"
        "  [cyan]1[/cyan]) Use the bundled sample resume\n"
        "  [cyan]2[/cyan]) Paste resume text\n"
        "  [cyan]3[/cyan]) Load resume from a file path"
    )
    choice = input("Your choice [1/2/3] (default 1): ").strip() or "1"
    if choice == "1":
        if not SAMPLE_RESUME_PATH.exists():
            console.print(f"[red]Sample resume not found at {SAMPLE_RESUME_PATH}. Falling back to paste.[/red]")
            return _read_multiline("Paste your resume text:")
        text = SAMPLE_RESUME_PATH.read_text()
        console.print(f"[green]Loaded sample resume ({len(text)} chars).[/green]")
        return text
    if choice == "2":
        return _read_multiline("Paste your resume text:")
    if choice == "3":
        path = input("Enter full path to resume file: ").strip().strip('"').strip("'")
        p = Path(path).expanduser()
        if not p.exists():
            console.print(f"[red]File not found: {p}[/red]")
            sys.exit(1)
        return p.read_text()
    console.print("[red]Invalid choice.[/red]")
    sys.exit(1)


def get_target_role() -> str:
    console.print(Rule("[bold]Step 2: Target role[/bold]"))
    role = input(
        "Enter the target job role (e.g. 'Senior Backend Engineer at a fintech startup')\n"
        "or press Enter to skip: "
    ).strip()
    return role


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Resume Reviewer & Career Coach - a LangChain + LangGraph multi-agent system.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in offline mock mode using canned LLM responses (no API key needed).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a resume file. If omitted, you will be prompted interactively.",
    )
    parser.add_argument(
        "--role",
        type=str,
        default=None,
        help="Target job role. If omitted, you will be prompted interactively.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip all prompts; requires --resume (or uses the bundled sample_resume.txt).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point — accepts dynamic user input and runs the multi-agent graph."""
    args = _parse_cli_args()
    if args.mock or os.getenv("DEMO_MOCK") == "1":
        _enable_mock_mode()

    banner = textwrap.dedent(
        """
        AI Resume Reviewer & Career Coach
        Multi-Agent System (LangChain + LangGraph)
        """
    ).strip()
    if MOCK_MODE:
        banner += "\n(running in MOCK mode - canned responses, no API calls)"
    else:
        provider = _detect_provider()
        model_name = {
            "gemini": GEMINI_MODEL,
            "groq": GROQ_MODEL,
            "openai": OPENAI_MODEL,
        }[provider]
        banner += f"\nProvider: {provider}  |  Model: {model_name}"
    console.print(Panel.fit(banner, border_style="bold cyan"))

    # Resume
    if args.resume:
        p = Path(args.resume).expanduser()
        if not p.exists():
            console.print(f"[red]Resume file not found: {p}[/red]")
            sys.exit(1)
        resume_text = p.read_text()
        console.print(f"[green]Loaded resume from {p} ({len(resume_text)} chars).[/green]")
    elif args.non_interactive:
        if not SAMPLE_RESUME_PATH.exists():
            console.print("[red]--non-interactive requires --resume or a sample_resume.txt.[/red]")
            sys.exit(1)
        resume_text = SAMPLE_RESUME_PATH.read_text()
        console.print(f"[green]Using bundled sample resume ({len(resume_text)} chars).[/green]")
    else:
        resume_text = get_resume_text()

    # Target role
    if args.role is not None:
        target_role = args.role
        console.print(f"[green]Target role: {target_role or '(none)'}[/green]")
    elif args.non_interactive:
        target_role = ""
    else:
        target_role = get_target_role()

    if not resume_text:
        console.print("[red]No resume provided. Exiting.[/red]")
        sys.exit(1)

    initial_state: ResumeState = {
        "resume_text": resume_text,
        "target_role": target_role,
        "trace": [],
    }

    console.print(Rule("[bold]Running the agent graph[/bold]"))
    app = build_graph()
    try:
        final_state: ResumeState = app.invoke(initial_state)
    except Exception as e:  # pragma: no cover
        msg = str(e)
        if "insufficient_quota" in msg or "RateLimitError" in type(e).__name__:
            console.print(
                Panel(
                    "[bold red]OpenAI returned insufficient_quota (HTTP 429).[/bold red]\n\n"
                    "Your API key works, but the account has no available credits.\n\n"
                    "Fix options:\n"
                    "  1. Add a payment method / credits at\n"
                    "     [cyan]https://platform.openai.com/account/billing[/cyan]\n"
                    "  2. Use a different key from an account that has credits.\n"
                    "  3. For now, re-run with the offline mock pipeline:\n"
                    "     [green]python multi_agent_system.py --mock[/green]",
                    title="API quota exhausted",
                    border_style="red",
                )
            )
            sys.exit(2)
        if "invalid_api_key" in msg or "Incorrect API key" in msg:
            console.print(
                Panel(
                    "[bold red]OpenAI rejected the API key.[/bold red]\n"
                    "Double-check [cyan].env[/cyan] contains a valid OPENAI_API_KEY.",
                    title="Invalid API key",
                    border_style="red",
                )
            )
            sys.exit(2)
        # Unknown error - re-raise so we still see the traceback
        raise

    console.print(Rule("[bold cyan]Final report[/bold cyan]"))
    report = final_state.get("final_report", "(no report produced)")
    console.print(Markdown(report))

    out_path = Path(__file__).parent / "resume_review_output.md"
    out_path.write_text(report)
    console.print(
        Panel.fit(
            f"[green]Report saved to:[/green] {out_path}",
            border_style="green",
        )
    )

    console.print(Rule("[bold]Agent trace[/bold]"))
    for i, step in enumerate(final_state.get("trace", []), 1):
        console.print(f"  [dim]{i}.[/dim] {step}")


if __name__ == "__main__":
    main()
