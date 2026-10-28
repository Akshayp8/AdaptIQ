"""
Pytest tests for the Adaptive AI Learning System.

Each test has a one-line docstring describing what it checks.
Run from the project root: pytest tests/
"""

from __future__ import annotations

import os
import pathlib
import sys

import pandas as pd
import pytest

# Put src/ on the path so imports work
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

# Change working directory to the project root so relative paths (data/, models/) work
os.chdir(pathlib.Path(__file__).parent.parent)

from adaptive_engine import db, fuzzy
from adaptive_engine.mastery import MasteryTracker
from adaptive_engine.engine import AdaptiveEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def use_temp_db(tmp_path, monkeypatch):
    """Redirect the database to a temporary file for every test."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.initialise_db()


# ---------------------------------------------------------------------------
# Test 1: Mastery rises after a correct answer
# ---------------------------------------------------------------------------

def test_mastery_rises_after_correct_answer():
    """Mastery probability for a topic should increase after a correct answer."""
    tracker = MasteryTracker()
    before  = tracker.get("Functions")
    tracker.update("Functions", correct=True)
    after   = tracker.get("Functions")
    assert after > before, "Mastery should rise after a correct answer"


# ---------------------------------------------------------------------------
# Test 2: Mastery falls after a wrong answer
# ---------------------------------------------------------------------------

def test_mastery_falls_after_wrong_answer():
    """Mastery probability for a topic should decrease after a wrong answer."""
    tracker = MasteryTracker()
    before  = tracker.get("Functions")
    tracker.update("Functions", correct=False)
    after   = tracker.get("Functions")
    assert after < before, "Mastery should fall after a wrong answer"


# ---------------------------------------------------------------------------
# Test 3: Mastery stays between 0 and 1
# ---------------------------------------------------------------------------

def test_mastery_stays_in_bounds():
    """Mastery probability must always stay between 0.0 and 1.0."""
    tracker = MasteryTracker()
    for _ in range(30):
        tracker.update("Loops and Conditionals", correct=True)
    for _ in range(30):
        tracker.update("Loops and Conditionals", correct=False)
    prob = tracker.get("Loops and Conditionals")
    assert 0.0 <= prob <= 1.0, f"Mastery out of bounds: {prob}"


# ---------------------------------------------------------------------------
# Test 4: Fuzzy level is higher for accurate+fast vs inaccurate+slow
# ---------------------------------------------------------------------------

def test_fuzzy_level_higher_for_good_performance():
    """A student who is accurate and fast should get a higher fuzzy level than one who is not."""
    level_good = fuzzy.compute_level(accuracy=0.9, speed=0.9)
    level_poor = fuzzy.compute_level(accuracy=0.1, speed=0.1)
    assert level_good > level_poor, (
        f"Good performance ({level_good}) should score higher than poor ({level_poor})"
    )


# ---------------------------------------------------------------------------
# Test 5: level_to_difficulty works at the boundaries
# ---------------------------------------------------------------------------

def test_level_to_difficulty_boundaries():
    """level_to_difficulty should return Easy for low levels and Hard for high levels."""
    assert fuzzy.level_to_difficulty(0.0)  == "Easy"
    assert fuzzy.level_to_difficulty(3.4)  == "Easy"
    assert fuzzy.level_to_difficulty(3.5)  == "Medium"
    assert fuzzy.level_to_difficulty(6.4)  == "Medium"
    assert fuzzy.level_to_difficulty(6.5)  == "Hard"
    assert fuzzy.level_to_difficulty(10.0) == "Hard"


# ---------------------------------------------------------------------------
# Test 6: Engine never repeats a question in one session
# ---------------------------------------------------------------------------

def test_engine_never_repeats_question():
    """The engine should return each question at most once per session."""
    engine = AdaptiveEngine(student_id=1)
    seen_ids: set[int] = set()
    for _ in range(20):
        q = engine.next_question()
        if q is None:
            break
        assert q["id"] not in seen_ids, f"Question {q['id']} was repeated!"
        seen_ids.add(q["id"])
        engine.record_answer(q["id"], correct=True, time_taken=15.0)


# ---------------------------------------------------------------------------
# Test 7: weak_topics reflects wrong answers
# ---------------------------------------------------------------------------

def test_weak_topics_reflects_wrong_answers():
    """After many wrong answers on a topic, it should appear in weak_topics."""
    tracker = MasteryTracker()
    for _ in range(10):
        tracker.update("Variables and Data Types", correct=False)
    weak = tracker.weak_topics()
    assert "Variables and Data Types" in weak, (
        "Topic with many wrong answers should be listed as weak"
    )


# ---------------------------------------------------------------------------
# Test 8: Classifier predicts a valid topic
# ---------------------------------------------------------------------------

def test_classifier_predicts_valid_topic():
    """The classifier should return one of the four known topics."""
    from adaptive_engine.classifier import predict_topic

    valid_topics = {
        "Variables and Data Types",
        "Loops and Conditionals",
        "Functions",
        "Lists and Dictionaries",
    }
    result = predict_topic("How do you define a function in Python?")
    assert result in valid_topics, f"Unexpected topic predicted: {result}"


# ---------------------------------------------------------------------------
# Test 9: questions.csv has 80 rows, 4 topics, 20 per topic, valid answers
# ---------------------------------------------------------------------------

def test_questions_csv_structure():
    """questions.csv should have 80 rows, 4 topics, 20 questions each, valid answers."""
    df = pd.read_csv(pathlib.Path("data") / "questions.csv")

    assert len(df) == 80, f"Expected 80 questions, got {len(df)}"

    topic_counts = df["topic"].value_counts()
    assert len(topic_counts) == 4, f"Expected 4 topics, got {len(topic_counts)}"
    for topic, count in topic_counts.items():
        assert count == 20, f"Topic '{topic}' has {count} questions (expected 20)"

    valid_answers = {"a", "b", "c", "d"}
    invalid = df[~df["answer"].isin(valid_answers)]
    assert invalid.empty, f"Invalid answer values found: {invalid['answer'].unique()}"


# ---------------------------------------------------------------------------
# Test 10: Login works with correct credentials, fails with wrong ones
# ---------------------------------------------------------------------------

def test_login_correct_and_wrong():
    """Login should succeed with correct credentials and fail with wrong password or ID."""
    # Correct credentials
    result = db.check_login(student_id=1, password="student123")
    assert result is not None, "Login should succeed with correct credentials"
    assert result["student_id"] == 1

    # Wrong password
    result_bad_pw = db.check_login(student_id=1, password="wrongpass")
    assert result_bad_pw is None, "Login should fail with wrong password"

    # Non-existent student ID
    result_bad_id = db.check_login(student_id=999, password="student123")
    assert result_bad_id is None, "Login should fail with a non-existent student ID"


# ---------------------------------------------------------------------------
# Test 11: Stored password is not plain text
# ---------------------------------------------------------------------------

def test_password_not_stored_as_plain_text():
    """The password stored in the database should be a SHA-256 hash, not plain text."""
    conn = db.get_connection()
    row  = conn.execute(
        "SELECT password FROM students WHERE student_id = 1"
    ).fetchone()
    conn.close()

    stored_password = row["password"]
    assert stored_password != "student123", "Password must not be stored as plain text"
    # SHA-256 hash is always 64 hex characters
    assert len(stored_password) == 64, (
        f"Stored password should be a 64-char SHA-256 hash, got length {len(stored_password)}"
    )
