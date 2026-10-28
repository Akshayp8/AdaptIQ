"""
Streamlit application for the Adaptive AI Learning System.

Pages controlled by st.session_state:
  - "login"   : Student authentication
  - "quiz"    : Adaptive quiz dashboard with live engine telemetry
  - "profile" : Student profile & analytics (score, radar chart, performance analysis, mistake history)
  - "results" : End-of-quiz summary
"""

from __future__ import annotations

import pathlib
import sys
import time
import textwrap
import importlib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure src/ is on the Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import adaptive_engine.db
import adaptive_engine.fuzzy
import adaptive_engine.mastery
import adaptive_engine.engine

importlib.reload(adaptive_engine.db)
importlib.reload(adaptive_engine.fuzzy)
importlib.reload(adaptive_engine.mastery)
importlib.reload(adaptive_engine.engine)

from adaptive_engine import db
from adaptive_engine.mastery import MasteryTracker
from adaptive_engine.engine import AdaptiveEngine, ALL_TOPICS

# ------------------------------------------------------------------
# Page configuration (must be the FIRST Streamlit call)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Adaptive Learning AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Load question bank once for lookup in history
# ------------------------------------------------------------------
QUESTIONS_CSV = pathlib.Path("data") / "questions.csv"
if QUESTIONS_CSV.exists():
    _QUESTIONS_DF = pd.read_csv(QUESTIONS_CSV)
    _QUESTION_MAP = {
        int(r["id"]): {
            "question": str(r["question"]),
            "topic": str(r["topic"]),
            "difficulty": str(r["difficulty"]),
            "answer": str(r["answer"]),
        }
        for _, r in _QUESTIONS_DF.iterrows()
    }
else:
    _QUESTION_MAP = {}

# ------------------------------------------------------------------
# Custom Dark Theme CSS
# ------------------------------------------------------------------
CUSTOM_CSS = textwrap.dedent("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, p, div:not([data-testid="stIcon"]), button, input, label, select, h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Preserve Streamlit icon ligatures & expander arrows */
[data-testid="stIcon"], .material-symbols-rounded, .material-symbols-sharp, .material-symbols-outlined, [data-testid="stExpander"] summary span svg {
    font-family: inherit !important;
}
details[data-testid="stExpander"] summary {
    color: #cbd5e1 !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}

/* Page Background */
.stApp {
    background-color: #080c16;
    color: #f1f5f9;
}

/* Navbar */
.nav-brand {
    font-size: 2.35rem;
    font-weight: 800;
    color: #c084fc;
    letter-spacing: -0.025em;
    display: flex;
    align-items: center;
    line-height: 1.15;
}
.nav-brand span {
    color: #818cf8;
}

/* Titles */
.main-title {
    color: #c084fc;
    font-size: 2.3rem;
    font-weight: 800;
    text-align: center;
    letter-spacing: -0.02em;
    margin-top: 3vh;
    margin-bottom: 0.35rem;
}
.main-title span {
    color: #818cf8;
}
.sub-title {
    color: #94a3b8;
    font-size: 1rem;
    text-align: center;
    margin-bottom: 2.2rem;
}

/* Profile Header */
.profile-header-title {
    font-size: 1.85rem;
    font-weight: 700;
    color: #f8fafc;
    display: inline-flex;
    align-items: center;
    gap: 12px;
}
.profile-badge {
    background: rgba(139, 92, 246, 0.22);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.45);
    border-radius: 9999px;
    padding: 3px 12px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.profile-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 0.2rem;
    margin-bottom: 1.5rem;
}

/* 4 KPI Cards in a row */
.kpi-card {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    min-height: 110px;
}
.kpi-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #94a3b8;
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 1.95rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.25rem;
}
.kpi-caption {
    font-size: 0.75rem;
    color: #64748b;
}

/* Mistake Cards */
.history-item-card {
    background-color: #0d1322;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    color: #cbd5e1;
    font-size: 0.92rem;
    line-height: 1.6;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

/* Question card on Quiz page */
.question-card {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-left: 6px solid #8b5cf6;
    border-radius: 14px;
    padding: 1.8rem 2.2rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    margin-bottom: 1.5rem;
}
.question-text {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 0.85rem;
    line-height: 1.55;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 9999px;
    font-size: 0.74rem;
    font-weight: 600;
    margin-right: 6px;
}
.badge-easy { background-color: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
.badge-medium { background-color: #78350f; color: #fde68a; border: 1px solid #b45309; }
.badge-hard { background-color: #7f1d1d; color: #fca5a5; border: 1px solid #b91c1c; }
.badge-topic { background-color: #2e1065; color: #d8b4fe; border: 1px solid #6b21a8; }

.badge-weak { background-color: #7f1d1d; color: #fca5a5; border: 1px solid #b91c1c; }
.badge-moderate { background-color: #78350f; color: #fde68a; border: 1px solid #b45309; }
.badge-strong { background-color: #064e3b; color: #6ee7b7; border: 1px solid #059669; }

/* Feedback */
.feedback-correct {
    background-color: #052e16;
    border: 1px solid #15803d;
    color: #86efac;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    font-weight: 600;
}
.feedback-wrong {
    background-color: #450a0a;
    border: 1px solid #991b1b;
    color: #fca5a5;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    font-weight: 600;
}

/* Primary buttons */
div.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%);
    color: #ffffff;
    border: none;
    border-radius: 9px;
    padding: 0.52rem 1.6rem;
    font-weight: 600;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);
    transition: all 0.2s ease-in-out;
}
div.stButton > button:hover {
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
    transform: translateY(-1px);
    color: #ffffff;
}

/* Containers with border styling */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25) !important;
    padding: 1.25rem !important;
}

/* Radio widget wrapper */
div[data-testid="stRadio"] {
    width: 100% !important;
}

/* Strictly hide any widget label for radio (removes "Options:") */
div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] label[data-testid="stWidgetLabel"],
div[data-testid="stRadio"] [data-testid="stWidgetLabel"] {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    visibility: hidden !important;
}

/* Clean, unbloated radio options list with comfortable spacing */
div[data-testid="stRadio"] div[role="radiogroup"],
div[data-testid="stRadio"] [data-testid="stRadioGroup"] {
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
}

div[data-testid="stRadio"] label,
div[data-testid="stRadioOption"],
[data-testid="stRadioOption"],
label[data-baseweb="radio"],
div[role="radiogroup"] label {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    padding: 6px 10px !important;
    min-height: auto !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
    transition: all 0.15s ease-in-out !important;
    margin: 0 !important;
}

div[data-testid="stRadio"] label:hover,
div[data-testid="stRadioOption"]:hover {
    background: rgba(139, 92, 246, 0.08) !important;
}

/* Option typography: shorter, perfectly clean and legible */
div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"],
div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span:not([data-testid="stRadioIcon"]) {
    font-size: 1.12rem !important;
    font-weight: 500 !important;
    line-height: 1.5 !important;
    color: #f1f5f9 !important;
    margin: 0 !important;
}

/* Radio circle icon */
div[data-testid="stRadio"] label [data-testid="stRadioIcon"],
div[data-testid="stRadio"] label div:first-child {
    transform: scale(1.15) !important;
    margin-right: 14px !important;
    flex-shrink: 0 !important;
}
</style>
""").strip()

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Initialise the database on startup
# ------------------------------------------------------------------
db.initialise_db()

# ------------------------------------------------------------------
# Session state defaults
# ------------------------------------------------------------------
def init_state() -> None:
    """Set up all session state keys with safe default values."""
    defaults = {
        "page":             "login",
        "student":          None,      # dict with student_id and name
        "engine":           None,      # AdaptiveEngine instance
        "current_q":        None,      # current question dict
        "q_number":         0,         # how many questions shown so far
        "quiz_length":      15,        # fixed default: 15 questions
        "submitted":        False,     # has current answer been submitted?
        "selected_option":  None,      # option picked (a/b/c/d)
        "feedback_correct": None,      # True / False after submission
        "q_start_time":     None,      # time question appeared
        "quiz_done":        False,
        "summary":          None,
        "target_topic":     None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

# ------------------------------------------------------------------
# Navigation helpers
# ------------------------------------------------------------------
def go_to(page: str) -> None:
    """Switch page via session state."""
    st.session_state["page"] = page

def logout() -> None:
    """Log out student and reset state."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()

def difficulty_badge(difficulty: str) -> str:
    """Return an HTML badge for a difficulty level."""
    css_class = f"badge-{difficulty.lower()}"
    return f'<span class="badge {css_class}">{difficulty}</span>'

def topic_badge(topic: str) -> str:
    """Return an HTML badge for a topic name."""
    return f'<span class="badge badge-topic">{topic}</span>'

def level_badge(label: str, score: float) -> str:
    """Return an HTML badge for a fuzzy level label."""
    css_class = f"badge-{label}"
    return f'<span class="badge {css_class}">{label.capitalize()} ({score:.1f}/10)</span>'

def start_quiz(target_topic: str | None = None, length: int = 15) -> None:
    """Start or restart a quiz session, optionally focused on a target topic."""
    student = st.session_state["student"]
    # Clear any lingering question radio selections
    for key in list(st.session_state.keys()):
        if key.startswith("radio_q_"):
            del st.session_state[key]

    engine  = AdaptiveEngine(student_id=student["student_id"], target_topic=target_topic)
    st.session_state["engine"]           = engine
    st.session_state["q_number"]         = 0
    st.session_state["quiz_done"]        = False
    st.session_state["summary"]          = None
    st.session_state["quiz_length"]      = length
    st.session_state["target_topic"]     = target_topic
    st.session_state["submitted"]        = False
    st.session_state["selected_option"]  = None
    st.session_state["feedback_correct"] = None
    load_next_question()

def load_next_question() -> None:
    """Load the next question from the adaptive engine."""
    engine = st.session_state["engine"]
    q = engine.next_question()
    if q is None or st.session_state["q_number"] >= st.session_state["quiz_length"]:
        finish_quiz()
        return
    st.session_state["current_q"]        = q
    st.session_state["submitted"]        = False
    st.session_state["selected_option"]  = None
    st.session_state["feedback_correct"] = None
    st.session_state["q_start_time"]     = time.time()

def finish_quiz() -> None:
    """Finalise the quiz and redirect to results."""
    engine = st.session_state["engine"]
    st.session_state["quiz_done"] = True
    st.session_state["summary"]   = engine.summary()
    go_to("results")

# ------------------------------------------------------------------
# Top Navigation Bar (Adaptive Learning AI — Exactly 3 Buttons)
# ------------------------------------------------------------------
def render_navbar(active_page: str) -> None:
    """
    Renders top navigation with brand and exactly 3 buttons:
    Left: Adaptive Learning AI
    Right: Quiz, Profile, Logout buttons
    """
    col_logo, col_buttons = st.columns([6, 4], vertical_alignment="center")
    with col_logo:
        st.markdown(
            '<div class="nav-brand">Adaptive Learning&nbsp;<span>AI</span></div>',
            unsafe_allow_html=True,
        )
    with col_buttons:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Quiz", use_container_width=True):
                # Start a fresh quiz if none active, if previous quiz finished, or if on results
                if (
                    st.session_state.get("engine") is None
                    or st.session_state.get("quiz_done")
                    or active_page == "results"
                ):
                    start_quiz()
                go_to("quiz")
                st.rerun()
        with c2:
            if st.button("Profile", use_container_width=True):
                go_to("profile")
                st.rerun()
        with c3:
            if st.button("Logout", use_container_width=True):
                logout()
                st.rerun()
    st.divider()

# ------------------------------------------------------------------
# Polished Vector SVG Radar Chart Generator
# ------------------------------------------------------------------
def render_svg_radar(mastery_dict: dict[str, float]) -> str:
    """
    Renders an ultra-clean, vector-perfect SVG radar chart for 4 topics.
    Styled in dark violet & cyan on dark slate (#0f172a).
    Zero distortion, balanced 4-axis geometry, sharp retina typography.
    """
    width, height = 520, 360
    cx, cy = 260, 180
    radius = 115.0

    topics = [
        {"key": "Variables and Data Types", "label": "Variables"},
        {"key": "Functions",                "label": "Functions"},
        {"key": "Loops and Conditionals",   "label": "Loops"},
        {"key": "Lists and Dictionaries",   "label": "Lists & Dicts"},
    ]

    # Clamp values between 0.0 and 1.0
    values = [max(0.0, min(1.0, float(mastery_dict.get(t["key"], 0.0)))) for t in topics]
    has_data = any(v > 0 for v in values)

    svg_parts: list[str] = []
    svg_parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" height="{height}" style="max-width: 520px; display: block; margin: 0 auto; user-select: none;">'
    )

    # Defs: Gradients and drop-shadow glow
    svg_parts.append("""
    <defs>
      <linearGradient id="radarPolyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#c084fc" stop-opacity="0.45" />
        <stop offset="100%" stop-color="#818cf8" stop-opacity="0.22" />
      </linearGradient>
      <filter id="radarGlow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur stdDeviation="3.5" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
    </defs>
    """)

    # Background subtle radial disc
    svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius + 4}" fill="rgba(15, 23, 42, 0.5)" />')

    # Concentric dashed rings & scale labels: 20%, 40%, 60%, 80%, 100%
    ring_percents = [0.20, 0.40, 0.60, 0.80, 1.00]
    for r_pct in ring_percents:
        r_val = radius * r_pct
        stroke_color = "#334155" if r_pct == 1.0 else "#1e293b"
        stroke_width = "1.4" if r_pct == 1.0 else "1.0"
        dash = "none" if r_pct == 1.0 else "4,4"
        svg_parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r_val:.1f}" fill="none" '
            f'stroke="{stroke_color}" stroke-width="{stroke_width}" stroke-dasharray="{dash}" />'
        )

        # Scale label along 45-degree spoke (top-right)
        diag_x = cx + (r_val * 0.7071) + 4
        diag_y = cy - (r_val * 0.7071) + 3
        pct_text = f"{int(r_pct * 100)}%"
        svg_parts.append(
            f'<text x="{diag_x:.1f}" y="{diag_y:.1f}" fill="#64748b" font-size="9px" '
            f'font-family="\'Inter\', -apple-system, sans-serif" font-weight="500">{pct_text}</text>'
        )

    # Crosshairs (vertical & horizontal axis lines)
    svg_parts.append(
        f'<line x1="{cx}" y1="{cy - radius}" x2="{cx}" y2="{cy + radius}" '
        f'stroke="#334155" stroke-width="1.2" stroke-dasharray="3,3" />'
    )
    svg_parts.append(
        f'<line x1="{cx - radius}" y1="{cy}" x2="{cx + radius}" y2="{cy}" '
        f'stroke="#334155" stroke-width="1.2" stroke-dasharray="3,3" />'
    )

    # 4 Vertex coordinates:
    # 0 (Top: Variables):       (cx, cy - v * radius)
    # 1 (Right: Functions):     (cx + v * radius, cy)
    # 2 (Bottom: Loops):        (cx, cy + v * radius)
    # 3 (Left: Lists & Dicts):  (cx - v * radius, cy)
    p_top   = (cx, cy - values[0] * radius)
    p_right = (cx + values[1] * radius, cy)
    p_bot   = (cx, cy + values[2] * radius)
    p_left  = (cx - values[3] * radius, cy)
    pts = [p_top, p_right, p_bot, p_left]

    if has_data:
        pts_str = " ".join([f"{p[0]:.1f},{p[1]:.1f}" for p in pts])
        # Data polygon
        svg_parts.append(
            f'<polygon points="{pts_str}" fill="url(#radarPolyGrad)" stroke="#c084fc" '
            f'stroke-width="2.5" stroke-linejoin="round" filter="url(#radarGlow)" />'
        )

        # Node circles at vertices
        for i, (px, py) in enumerate(pts):
            if values[i] > 0:
                svg_parts.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.2" fill="#38bdf8" '
                    f'stroke="#ffffff" stroke-width="2" />'
                )
    else:
        # Clean empty state
        svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="4" fill="#475569" />')
        svg_parts.append(
            f'<text x="{cx}" y="{cy + 22}" text-anchor="middle" fill="#64748b" '
            f'font-size="11px" font-family="\'Inter\', -apple-system, sans-serif">No Quiz Activity</text>'
        )

    # Axis Labels with mastery callouts
    # Top: Variables
    pct_top = f" ({int(values[0] * 100)}%)" if values[0] > 0 else ""
    svg_parts.append(
        f'<text x="{cx}" y="{cy - radius - 16}" text-anchor="middle" '
        f'font-family="\'Inter\', -apple-system, sans-serif" font-size="13px" font-weight="700" fill="#f8fafc">'
        f'Variables<tspan fill="#38bdf8" font-size="11.5px" font-weight="600">{pct_top}</tspan>'
        f'</text>'
    )

    # Right: Functions
    pct_right = f" ({int(values[1] * 100)}%)" if values[1] > 0 else ""
    svg_parts.append(
        f'<text x="{cx + radius + 15}" y="{cy + 4}" text-anchor="start" '
        f'font-family="\'Inter\', -apple-system, sans-serif" font-size="13px" font-weight="700" fill="#f8fafc">'
        f'Functions<tspan fill="#38bdf8" font-size="11.5px" font-weight="600">{pct_right}</tspan>'
        f'</text>'
    )

    # Bottom: Loops
    pct_bot = f" ({int(values[2] * 100)}%)" if values[2] > 0 else ""
    svg_parts.append(
        f'<text x="{cx}" y="{cy + radius + 25}" text-anchor="middle" '
        f'font-family="\'Inter\', -apple-system, sans-serif" font-size="13px" font-weight="700" fill="#f8fafc">'
        f'Loops<tspan fill="#38bdf8" font-size="11.5px" font-weight="600">{pct_bot}</tspan>'
        f'</text>'
    )

    # Left: Lists & Dicts
    pct_left = f" ({int(values[3] * 100)}%)" if values[3] > 0 else ""
    svg_parts.append(
        f'<text x="{cx - radius - 15}" y="{cy + 4}" text-anchor="end" '
        f'font-family="\'Inter\', -apple-system, sans-serif" font-size="13px" font-weight="700" fill="#f8fafc">'
        f'Lists &amp; Dicts<tspan fill="#38bdf8" font-size="11.5px" font-weight="600">{pct_left}</tspan>'
        f'</text>'
    )

    svg_parts.append('</svg>')
    return "".join(svg_parts)

# ------------------------------------------------------------------
# PAGE 1: Login (Centered "Adaptive Learning AI", Centered "Sign In")
# ------------------------------------------------------------------
def render_login() -> None:
    """Show the dark login form centered on page."""
    st.markdown(
        '<div class="main-title">Adaptive Learning <span>AI</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-title">Personalised Python quizzes that adapt to your skill in real time</p>',
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 1.25, 1])
    with col:
        with st.container(border=True):
            st.markdown(
                '<h3 style="text-align: center; margin-top: 0.2rem; margin-bottom: 0.25rem; font-weight: 700;">Sign In</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p style="text-align: center; color: #94a3b8; font-size: 0.88rem; margin-bottom: 1.4rem;">'
                'Enter your student credentials to begin.'
                '</p>',
                unsafe_allow_html=True,
            )

            student_id_str = st.text_input("Student ID", placeholder="Enter your Student ID (e.g. 1)")
            password_input = st.text_input("Password", placeholder="Enter password")

            st.write("")
            submitted = st.button("Sign In →", use_container_width=True)

            if submitted:
                cleaned_id = student_id_str.strip()
                if not cleaned_id.isdigit():
                    st.error("Please enter a valid numeric Student ID.")
                else:
                    result = db.check_login(int(cleaned_id), password_input)
                    if result is None:
                        st.error("Invalid Student ID or password. Please try again.")
                    else:
                        st.session_state["student"]     = result
                        st.session_state["quiz_length"] = 15
                        go_to("profile")
                        st.rerun()

# ------------------------------------------------------------------
# PAGE 2: Student Profile & Analytics
# ------------------------------------------------------------------
def render_profile() -> None:
    """
    Renders the Profile screen:
    - Navbar with 3 buttons: Quiz, Profile, Logout
    - Header with Student Name & Student Profile badge
    - 4 KPI cards: Student Score, Weak Topics, Strong Topics, Mistakes Logged
    - Tabs: Overview (Radar + Performance Analysis aligned inside containers), Mistakes History, Cognitive Behaviour, Timeline
    """
    student = st.session_state["student"]
    attempts = db.get_attempts(student["student_id"])

    # 1. Navbar (3 buttons only)
    render_navbar("profile")

    # 2. Header row
    st.markdown(
        f'<div class="profile-header-title">'
        f'{student["name"]} <span class="profile-badge">Student Profile</span>'
        f'</div>'
        f'<div class="profile-subtitle">'
        f'Student ID: {student["student_id"]} — Competencies, mistake history, and cognitive pacing'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 3. Compute dynamic metrics based on student's actual database history
    total_attempts = len(attempts)
    wrong_attempts = [a for a in attempts if not a["correct"]]
    mistakes_count = len(wrong_attempts)

    # Topics that this student has ACTUALLY been tested on
    tested_topics = set(a["topic"] for a in attempts)

    mastery_map: dict[str, float] = {}
    topic_stats: dict[str, dict] = {}

    if total_attempts > 0:
        tracker = MasteryTracker()
        for a in attempts:
            tracker.update(a["topic"], bool(a["correct"]))

        for t in ALL_TOPICS:
            if t in tested_topics:
                m_val = round(tracker.get(t), 3)
                mastery_map[t] = m_val
                # Count correct / total for this topic
                t_attempts = [a for a in attempts if a["topic"] == t]
                t_correct = sum(1 for a in t_attempts if a["correct"])
                t_total = len(t_attempts)
                t_mistakes = t_total - t_correct
                topic_stats[t] = {
                    "mastery": m_val,
                    "correct": t_correct,
                    "total": t_total,
                    "mistakes": t_mistakes,
                    "accuracy": t_correct / t_total,
                }
            else:
                mastery_map[t] = 0.0

        weak_topics_list = [t for t in tested_topics if mastery_map[t] < 0.60]
        strong_topics_list = [t for t in tested_topics if mastery_map[t] >= 0.60]

        accuracy = sum(1 for a in attempts if a["correct"]) / total_attempts
        avg_tested_mastery = sum(mastery_map[t] for t in tested_topics) / len(tested_topics)
        score_val = int(avg_tested_mastery * 50 + accuracy * 50)
        score_display = f"{score_val}/100"
        score_caption = "Overall learning performance"
    else:
        for t in ALL_TOPICS:
            mastery_map[t] = 0.0
        weak_topics_list = []
        strong_topics_list = []
        score_display = "0/100"
        score_caption = "No quiz attempts yet"

    # 4 KPI Cards in a row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Student Score</div>'
            f'<div class="kpi-value" style="color:#c084fc;">{score_display}</div>'
            f'<div class="kpi-caption">{score_caption}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with kpi2:
        weak_caption = "Need attention (&lt;60%)" if total_attempts > 0 else "No weak topics recorded"
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Weak Topics</div>'
            f'<div class="kpi-value" style="color:#f87171;">{len(weak_topics_list)}</div>'
            f'<div class="kpi-caption">{weak_caption}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with kpi3:
        strong_caption = "Mastered (&ge;60%)" if total_attempts > 0 else "No topics evaluated yet"
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Strong Topics</div>'
            f'<div class="kpi-value" style="color:#4ade80;">{len(strong_topics_list)}</div>'
            f'<div class="kpi-caption">{strong_caption}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with kpi4:
        mistake_caption = "Recorded incorrect answers" if total_attempts > 0 else "No incorrect answers"
        st.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Mistakes Logged</div>'
            f'<div class="kpi-value" style="color:#fbbf24;">{mistakes_count}</div>'
            f'<div class="kpi-caption">{mistake_caption}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.write("")

    # 4. Tabs
    tab_overview, tab_mistakes, tab_behaviour, tab_timeline = st.tabs(
        ["Overview", "Mistakes History", "Cognitive Behaviour", "Timeline"]
    )

    # ---------------- TAB 1: OVERVIEW ----------------
    with tab_overview:
        col_left, col_right = st.columns([1, 1], gap="large")

        # Left Column: Polished SVG Radar Chart inside bordered card
        with col_left:
            with st.container(border=True):
                st.markdown(
                    '<div style="display:flex; align-items:center; gap:10px; margin-top:0.2rem; margin-bottom:0.25rem;">'
                    '<h4 style="margin:0; font-weight:700;">Topic Mastery Radar</h4>'
                    '<span class="profile-badge" style="font-size:0.72rem;">Bayesian Knowledge Tracing</span>'
                    '</div>'
                    '<p style="color:#64748b;font-size:0.84rem;margin-bottom:0.8rem;">Cognitive mastery probabilities modeled via BKT</p>',
                    unsafe_allow_html=True,
                )
                svg_radar = render_svg_radar(mastery_map)
                st.markdown(svg_radar, unsafe_allow_html=True)

                # BKT Model Parameters & Formula Inspection
                with st.expander("BKT Model Parameters & Formula"):
                    st.markdown(
                        """
                        <div style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.65;">
                          <p style="margin-bottom: 0.5rem;"><b style="color: #c084fc;">Bayesian Knowledge Tracing (BKT) Parameters:</b></p>
                          <ul style="padding-left: 1.2rem; margin-bottom: 0.75rem; color: #94a3b8;">
                            <li><b style="color: #f1f5f9;">Prior P(L₀) = 0.30:</b> Initial baseline probability of understanding.</li>
                            <li><b style="color: #f1f5f9;">Guess Rate (G) = 0.20:</b> Chance of correct answer by lucky guess (4-choice MCQ).</li>
                            <li><b style="color: #f1f5f9;">Slip Rate (S) = 0.10:</b> Chance of careless error despite knowing concept.</li>
                            <li><b style="color: #f1f5f9;">Learning Boost (T) = 0.15:</b> Cognitive transition gained per practice attempt.</li>
                            <li><b style="color: #f1f5f9;">Cromwell Bound = 0.95:</b> Upper probability cap preventing absorbing states so mistakes always penalize.</li>
                          </ul>
                          <div style="padding: 0.5rem 0.75rem; background: #080c16; border: 1px solid #1e293b; border-radius: 6px; font-size: 0.78rem; color: #64748b;">
                            <b>Inference Rule:</b> Correct answer yields <i>P(L|Correct) + 0.15</i>; wrong answer yields <i>P(L|Wrong)</i> penalizing mastery.
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if total_attempts == 0:
                    st.caption("No evaluation data yet. Complete questions in the Quiz tab to calibrate your radar.")

        # Right Column: Performance Analysis (No "AI" in title)
        with col_right:
            with st.container(border=True):
                st.markdown(
                    '<h4 style="margin-top:0.2rem;margin-bottom:0.2rem;font-weight:700;">Performance Analysis</h4>'
                    '<p style="color:#64748b;font-size:0.84rem;margin-bottom:1rem;">Topic-by-topic competency and error breakdown</p>',
                    unsafe_allow_html=True,
                )

                if total_attempts == 0:
                    st.info(f"No quiz data recorded for {student['name']} yet. Click below to begin.")
                    st.write("")
                    if st.button("Start Your First Quiz →", use_container_width=True):
                        start_quiz()
                        go_to("quiz")
                        st.rerun()
                else:
                    st.markdown(
                        '<div style="color:#cbd5e1;line-height:1.7;font-size:0.92rem;">'
                        '<h5 style="color:#f8fafc;font-size:0.98rem;margin-top:0.6rem;margin-bottom:0.4rem;">Topic Competency Breakdown:</h5>',
                        unsafe_allow_html=True,
                    )

                    # Transparently display every evaluated topic with real numbers
                    for topic in tested_topics:
                        stat = topic_stats[topic]
                        m_pct = int(stat["mastery"] * 100)
                        c_count = stat["correct"]
                        tot = stat["total"]
                        m_count = stat["mistakes"]

                        if stat["mastery"] >= 0.80:
                            status_badge = f'<span style="color:#4ade80;font-weight:600;">Mastered ({m_pct}%)</span>'
                        elif stat["mastery"] >= 0.60:
                            status_badge = f'<span style="color:#38bdf8;font-weight:600;">Proficient ({m_pct}%)</span>'
                        else:
                            status_badge = f'<span style="color:#f87171;font-weight:600;">Needs Practice ({m_pct}%)</span>'

                        detail_err = f" — {m_count} error(s)" if m_count > 0 else " — 0 errors"
                        st.markdown(
                            f'<p style="margin-bottom:0.45rem;">'
                            f'• <b>{topic}:</b> {status_badge} &nbsp;({c_count}/{tot} correct{detail_err})'
                            f'</p>',
                            unsafe_allow_html=True,
                        )

                    untested = [t for t in ALL_TOPICS if t not in tested_topics]
                    if untested:
                        st.markdown(
                            f'<p style="color:#94a3b8;margin-top:0.4rem;margin-bottom:0.4rem;">'
                            f'• <b>Pending Evaluation:</b> {", ".join(untested)} (Not tested yet).'
                            f'</p>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        '<h5 style="color:#f8fafc;font-size:0.98rem;margin-top:1.2rem;margin-bottom:0.3rem;">Mistake Breakdown:</h5>',
                        unsafe_allow_html=True,
                    )

                    if mistakes_count > 0:
                        # Find which specific topics had mistakes
                        error_topics = [t for t in tested_topics if topic_stats[t]["mistakes"] > 0]
                        error_summary = ", ".join([f"{t} ({topic_stats[t]['mistakes']} error{'s' if topic_stats[t]['mistakes'] > 1 else ''})" for t in error_topics])

                        st.markdown(
                            f'<p style="margin-bottom:0.4rem;">'
                            f'• <b>Logged Mistakes:</b> {mistakes_count} incorrect answer(s) across: <b>{error_summary}</b>.'
                            f'</p>'
                            f'<p style="margin-bottom:0.4rem;">'
                            f'• <b>Action:</b> Review the <i>Mistakes History</i> tab to inspect the exact questions and correct answers.'
                            f'</p>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<p style="color:#86efac;">• Flawless accuracy! Zero incorrect answers logged across all attempted questions.</p>',
                            unsafe_allow_html=True,
                        )

                    # Dynamic Remedial Practice Recommendation
                    weak_candidates = sorted(
                        tested_topics,
                        key=lambda t: (topic_stats[t]["mastery"], -topic_stats[t]["mistakes"])
                    )
                    if weak_candidates:
                        weakest_t = weak_candidates[0]
                        w_stat = topic_stats[weakest_t]
                        w_pct = int(w_stat["mastery"] * 100)
                        w_err = w_stat["mistakes"]
                        if w_stat["mastery"] < 0.85 or w_err > 0:
                            st.markdown(
                                f'<div style="background: rgba(139, 92, 246, 0.12); border: 1px solid rgba(168, 85, 247, 0.35); '
                                f'border-radius: 10px; padding: 0.9rem 1.1rem; margin-top: 1rem; margin-bottom: 0.8rem;">'
                                f'<div style="color: #c084fc; font-weight: 700; font-size: 0.88rem; margin-bottom: 0.2rem;">🎯 Targeted Remedial Action</div>'
                                f'<div style="color: #cbd5e1; font-size: 0.83rem; line-height: 1.5;">'
                                f'Prioritize practice on <b>{weakest_t}</b> ({w_pct}% mastery, {w_err} mistake{"s" if w_err != 1 else ""} logged). '
                                f'Launch a 5-question targeted adaptive drill to strengthen this module.'
                                f'</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            if st.button(f"Start Targeted Practice ({weakest_t}) →", use_container_width=True, key=f"btn_remedial_{weakest_t}"):
                                start_quiz(target_topic=weakest_t, length=5)
                                go_to("quiz")
                                st.rerun()

                    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- TAB 2: MISTAKES HISTORY ----------------
    with tab_mistakes:
        st.markdown(
            '<h3 style="color:#38bdf8;margin-top:0.5rem;margin-bottom:0.3rem;">Quiz Participation History</h3>'
            '<p style="color:#94a3b8;font-size:0.9rem;margin-bottom:1.5rem;">'
            'Detailed log of recorded struggle points and concept errors'
            '</p>',
            unsafe_allow_html=True,
        )

        if not wrong_attempts:
            st.info(f"No recorded mistakes for {student['name']}. Answer quiz questions to populate this log.")
        else:
            for attempt in reversed(wrong_attempts):
                qid = attempt["question_id"]
                q_info = _QUESTION_MAP.get(qid, {})
                q_text = q_info.get("question", f"Question #{qid}")
                topic_name = attempt["topic"]
                t_taken = attempt["time_taken"]
                timestamp_str = attempt.get("timestamp", "")
                try:
                    dt = datetime.fromisoformat(timestamp_str)
                    formatted_date = dt.strftime("%A, %B %d, %Y")
                except Exception:
                    formatted_date = "Recently"

                st.markdown(
                    f'<div class="history-item-card">'
                    f'User {student["name"]} struggled with the topic <b>\'{topic_name}\'</b>, '
                    f'specifically with question: <i>"{q_text}"</i>. | '
                    f'When: {formatted_date} | Involving: {student["name"]} (ID: {student["student_id"]}) | '
                    f'The user answered incorrectly after {t_taken:.1f}s at {attempt["difficulty"]} difficulty tier.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ---------------- TAB 3: COGNITIVE BEHAVIOUR ----------------
    with tab_behaviour:
        st.markdown("### Cognitive Behaviour & Pacing")
        st.caption("Pacing and accuracy metrics calibrated by the adaptive fuzzy logic engine.")
        st.write("")

        if attempts:
            df_att = pd.DataFrame(attempts)
            b1, b2, b3 = st.columns(3)
            with b1:
                easy_time = df_att[df_att["difficulty"] == "Easy"]["time_taken"].mean()
                st.metric("Avg Time (Easy)", f"{easy_time:.1f} s" if not np.isnan(easy_time) else "N/A")
            with b2:
                med_time = df_att[df_att["difficulty"] == "Medium"]["time_taken"].mean()
                st.metric("Avg Time (Medium)", f"{med_time:.1f} s" if not np.isnan(med_time) else "N/A")
            with b3:
                hard_time = df_att[df_att["difficulty"] == "Hard"]["time_taken"].mean()
                st.metric("Avg Time (Hard)", f"{hard_time:.1f} s" if not np.isnan(hard_time) else "N/A")

            st.write("")

            # Cognitive Pacing Quadrant Matrix
            avg_time = float(df_att["time_taken"].mean())
            overall_acc = float(df_att["correct"].mean())

            if overall_acc >= 0.70 and avg_time <= 30.0:
                q_title = "Quadrant 1: Fast & Accurate (Fluency / Mastered)"
                q_badge = "Fluency Achieved"
                q_color = "#4ade80"
                q_desc = "Demonstrates high automaticity and rapid syntax recall. Concepts are deeply internalized."
            elif overall_acc >= 0.70 and avg_time > 30.0:
                q_title = "Quadrant 2: Methodical & Deliberate (High Precision)"
                q_badge = "High Precision"
                q_color = "#38bdf8"
                q_desc = "Takes time to carefully trace variable state and logic before answering. High accuracy with minimal slip risk."
            elif overall_acc < 0.70 and avg_time <= 30.0:
                q_title = "Quadrant 3: Impulsive / Rushing (High Slip Risk)"
                q_badge = "Caution: Rushing"
                q_color = "#fbbf24"
                q_desc = "Answers quickly but incurs avoidable misconceptions or slips. Recommend slowing down to trace conditionals and loop bounds."
            else:
                q_title = "Quadrant 4: High Cognitive Load (Needs Practice)"
                q_badge = "Needs Support"
                q_color = "#f87171"
                q_desc = "Requires extended deliberation while facing concept friction. Recommend focused practice on core syntax patterns."

            st.markdown(
                f'<div style="background: #0d1322; border: 1px solid #1e293b; border-radius: 12px; padding: 1.1rem 1.3rem; margin-bottom: 1.2rem;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem;">'
                f'<b style="color: #f1f5f9; font-size: 0.95rem;">{q_title}</b>'
                f'<span class="badge" style="background: {q_color}22; color: {q_color}; border: 1px solid {q_color}; font-size: 0.74rem;">{q_badge}</span>'
                f'</div>'
                f'<p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0; line-height: 1.5;">{q_desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("##### Response Time History")
            df_chart = df_att[["id", "time_taken"]].set_index("id")
            st.line_chart(df_chart, height=220)
        else:
            st.info(f"No response pacing data available yet for {student['name']}. Complete quiz questions to evaluate.")

    # ---------------- TAB 4: TIMELINE ----------------
    with tab_timeline:
        st.markdown("### Activity Timeline")
        st.caption("Chronological stream of all question attempts.")
        st.write("")

        if attempts:
            df_all = pd.DataFrame(attempts)[::-1]
            rows = []
            for _, a in df_all.iterrows():
                rows.append({
                    "Attempt ID": a["id"],
                    "Topic": a["topic"],
                    "Difficulty": a["difficulty"],
                    "Outcome": "Correct" if a["correct"] else "Incorrect",
                    "Time": f"{a['time_taken']:.1f}s",
                    "Timestamp": a["timestamp"][:19].replace("T", " "),
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            st.write("")
            csv_bytes = df_all.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Report (CSV)",
                data=csv_bytes,
                file_name=f"learning_report_student_{student['student_id']}.csv",
                mime="text/csv",
            )
        else:
            st.info(f"No quiz attempts recorded yet for {student['name']}.")

# ------------------------------------------------------------------
# PAGE 3: Quiz Dashboard
# ------------------------------------------------------------------
def render_quiz() -> None:
    """Show the current question, answer options, and live adaptation panel."""
    student   = st.session_state["student"]
    engine    = st.session_state["engine"]
    q         = st.session_state["current_q"]
    q_num     = st.session_state["q_number"]
    q_total   = st.session_state["quiz_length"]
    submitted = st.session_state["submitted"]

    # If previous quiz finished or engine/q missing, start a fresh quiz immediately
    if st.session_state.get("quiz_done") or engine is None or q is None:
        start_quiz()
        st.rerun()

    # Navbar (3 buttons only)
    render_navbar("quiz")

    # Progress bar
    progress = q_num / q_total if q_total else 0.0
    st.progress(progress, text=f"Question {q_num + 1} of {q_total}")
    st.write("")

    # Main Layout
    main_col, side_col = st.columns([2.8, 1.2], gap="large")

    with main_col:
        remedial_tag = ""
        if st.session_state.get("target_topic"):
            remedial_tag = f'<span class="badge" style="background:#3b0764; color:#e9d5ff; border:1px solid #7c3aed;">Targeted Drill: {st.session_state["target_topic"]}</span> '

        badges = remedial_tag + difficulty_badge(q["difficulty"]) + " " + topic_badge(q["topic"])
        st.markdown(
            f"""
            <div class="question-card">
              <div>{badges}</div>
              <div class="question-text">Q{q_num + 1}. {q["question"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        options = {
            "a": q["option_a"],
            "b": q["option_b"],
            "c": q["option_c"],
            "d": q["option_d"],
        }

        if not submitted:
            st.markdown("##### Select your answer:")
            selected = st.radio(
                "",
                options=list(options.keys()),
                format_func=lambda k: f"({k.upper()})  {options[k]}",
                index=None,
                key=f"radio_q_{q['id']}",
                label_visibility="collapsed",
            )
            st.session_state["selected_option"] = selected

            st.write("")
            if st.button("Submit Answer", disabled=(selected is None)):
                _submit_answer(q, selected, engine)
                st.rerun()
        else:
            st.markdown("##### Question Options:")
            for key, text in options.items():
                if key == q["answer"]:
                    badge_html = '<span style="float:right; color:#86efac; font-size:0.85rem; font-weight:600;">✓ Correct Answer</span>'
                    prefix = f"<b>({key.upper()})</b> {text}"
                    text_color = "#86efac"
                elif key == st.session_state["selected_option"]:
                    badge_html = '<span style="float:right; color:#fca5a5; font-size:0.85rem; font-weight:600;">✗ Your Choice</span>'
                    prefix = f"~~<b>({key.upper()})</b> {text}~~"
                    text_color = "#fca5a5"
                else:
                    badge_html = ""
                    prefix = f"<b>({key.upper()})</b> {text}"
                    text_color = "#cbd5e1"

                st.markdown(
                    f'<div style="padding: 6px 10px; font-size: 1.12rem; font-weight: 500; '
                    f'color: {text_color}; margin-bottom: 8px; display: flex; '
                    f'align-items: center; justify-content: space-between;">'
                    f'<span>{prefix}</span> {badge_html}</div>',
                    unsafe_allow_html=True,
                )

            correct_key  = q["answer"]
            correct_text = options[correct_key]
            if st.session_state["feedback_correct"]:
                st.markdown(
                    f'<div class="feedback-correct"><b>Correct:</b> The answer is '
                    f'({correct_key.upper()}) {correct_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="feedback-wrong"><b>Incorrect:</b> The correct answer is '
                    f'({correct_key.upper()}) {correct_text}</div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            if q_num + 1 >= q_total:
                if st.button("View Final Results →", use_container_width=True):
                    finish_quiz()
                    st.rerun()
            else:
                if st.button("Next Question →", use_container_width=True):
                    st.session_state["q_number"] += 1
                    load_next_question()
                    st.rerun()

    with side_col:
        with st.container(border=True):
            st.markdown("#### Live Telemetry")
            level_info = engine.get_level()
            label      = level_info["label"]
            score      = level_info["score"]
            next_diff  = level_info["difficulty"]

            st.markdown(
                f'<div style="margin-bottom:0.75rem;"><b>Estimated Level:</b><br>{level_badge(label, score)}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div style="margin-bottom:0.75rem;"><b>Next Difficulty:</b><br>{difficulty_badge(next_diff)}</div>',
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Accuracy", f"{level_info['accuracy']:.0%}")
            with m2:
                st.metric("Speed", f"{level_info['speed']:.0%}")

            acc_val = level_info["accuracy"]
            spd_val = level_info["speed"]
            acc_str = "High" if acc_val >= 0.70 else ("Med" if acc_val >= 0.40 else "Low")
            spd_str = "Fast" if spd_val >= 0.70 else ("Med" if spd_val >= 0.40 else "Slow")
            st.markdown(
                f'<div style="font-size:0.77rem; color:#94a3b8; margin-top:0.6rem; padding:0.55rem 0.75rem; background:#080c16; border: 1px solid #1e293b; border-radius:8px;">'
                f'<b style="color:#c084fc;">Active Mamdani Rule:</b><br>'
                f'{acc_str} Accuracy + {spd_str} Speed → <b style="color:#f1f5f9;">{next_diff} ({score:.1f}/10)</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.write("")
            if st.button("View Profile & Analytics", use_container_width=True):
                go_to("profile")
                st.rerun()


def _submit_answer(q: dict, selected: str, engine: AdaptiveEngine) -> None:
    """Record student's answer in database and engine."""
    time_taken = time.time() - (st.session_state["q_start_time"] or time.time())
    correct    = (selected == q["answer"])
    engine.record_answer(q["id"], correct, time_taken)
    st.session_state["submitted"]        = True
    st.session_state["feedback_correct"] = correct

# ------------------------------------------------------------------
# PAGE 4: Results
# ------------------------------------------------------------------
def render_results() -> None:
    """Show end of quiz summary."""
    student = st.session_state["student"]
    summary = st.session_state["summary"]

    render_navbar("results")

    if summary is None:
        st.info("No quiz results available yet.")
        if st.button("Go to Quiz"):
            start_quiz()
            go_to("quiz")
            st.rerun()
        return

    st.markdown(f"## Quiz Results — {student['name']}")
    st.caption("Session performance summary and adaptive recommendations.")
    st.write("")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Score", f"{summary['correct']} / {summary['total']}")
    with col2:
        st.metric("Accuracy", f"{summary['accuracy']:.0%}")
    with col3:
        st.metric("Avg Speed / Question", f"{summary['avg_time']:.1f} s")
    with col4:
        st.metric("Overall Level", summary['level']['label'].capitalize())

    st.write("")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        with st.container(border=True):
            st.markdown("### Topic Mastery")
            mastery = summary["topic_mastery"]
            df_m = pd.DataFrame({
                "Topic": list(mastery.keys()),
                "Mastery": list(mastery.values()),
            }).set_index("Topic")
            st.bar_chart(df_m, height=260)

    with chart_col2:
        with st.container(border=True):
            st.markdown("### Mastery Over Time")
            if summary["mastery_history"]:
                df_history = pd.DataFrame(summary["mastery_history"]).set_index("question")
                cols = [c for c in df_history.columns if df_history[c].notna().any()]
                if cols:
                    st.line_chart(df_history[cols], height=260)
            else:
                st.info("No progression history.")

    st.write("")
    act_col1, act_col2, act_col3 = st.columns(3)
    with act_col1:
        if st.button("Start New Quiz →", use_container_width=True):
            start_quiz()
            go_to("quiz")
            st.rerun()
    with act_col2:
        if st.button("View Profile & Analytics", use_container_width=True):
            go_to("profile")
            st.rerun()
    with act_col3:
        attempts = db.get_attempts(student["student_id"])
        if attempts:
            df_rep = pd.DataFrame(attempts)
            st.download_button(
                label="Download Report (CSV)",
                data=df_rep.to_csv(index=False).encode("utf-8"),
                file_name=f"learning_report_student_{student['student_id']}.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ------------------------------------------------------------------
# Page router
# ------------------------------------------------------------------
page = st.session_state["page"]

if page == "login":
    render_login()
elif page == "quiz":
    if st.session_state["student"] is None:
        go_to("login")
        st.rerun()
    else:
        render_quiz()
elif page == "profile":
    if st.session_state["student"] is None:
        go_to("login")
        st.rerun()
    else:
        render_profile()
elif page == "results":
    if st.session_state["student"] is None:
        go_to("login")
        st.rerun()
    else:
        render_results()
