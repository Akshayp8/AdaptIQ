"""
Adaptive Engine — the brain of the learning system.

This class ties together the fuzzy logic, Bayesian mastery tracker,
question bank, and database into one simple interface that the app uses.
"""

from __future__ import annotations

import pathlib
import random
from typing import Any

import pandas as pd

from adaptive_engine import db, fuzzy, mastery as mastery_mod

# Set a fixed random seed so results are reproducible
random.seed(42)

# All four topics in the question bank
ALL_TOPICS = [
    "Variables and Data Types",
    "Loops and Conditionals",
    "Functions",
    "Lists and Dictionaries",
]

# The first question is always Medium difficulty
FIRST_DIFFICULTY = "Medium"

# Probability of choosing the weakest topic (vs. a random topic)
WEAK_TOPIC_CHANCE = 0.70

# How many recent answers to use for the fuzzy calculation
WINDOW_SIZE = 5


class AdaptiveEngine:
    """
    Controls the flow of the quiz for one student.

    Usage:
        engine = AdaptiveEngine(student_id=5)
        question = engine.next_question()
        engine.record_answer(question["id"], correct=True, time_taken=18.0)
    """

    def __init__(
        self, student_id: int, target_topic: str | None = None
    ) -> None:
        # The student we are tracking
        self.student_id = student_id

        # Optional remedial target topic (focus mode)
        self.target_topic = target_topic

        # Load all questions from CSV once
        csv_path = pathlib.Path("data") / "questions.csv"
        self._all_questions = pd.read_csv(csv_path)

        # Bayesian mastery tracker (one per student session)
        self._mastery = mastery_mod.MasteryTracker()

        # List of question IDs already asked this session
        self._asked_ids: list[int] = []

        # History of (correct, speed_score) tuples for the fuzzy input
        self._answer_history: list[tuple[bool, float]] = []

    def record_answer(
        self, question_id: int, correct: bool, time_taken: float
    ) -> None:
        """
        Record one answer from the student.

        Saves to the database, updates the mastery tracker,
        and stores the answer in local history for the fuzzy engine.
        """
        # Look up topic and difficulty for this question
        row = self._all_questions[
            self._all_questions["id"] == question_id
        ].iloc[0]
        topic      = str(row["topic"])
        difficulty = str(row["difficulty"])

        # Save to database
        db.save_attempt(
            student_id  = self.student_id,
            question_id = question_id,
            topic       = topic,
            difficulty  = difficulty,
            correct     = correct,
            time_taken  = time_taken,
        )

        # Update Bayesian mastery for this topic
        self._mastery.update(topic, correct)

        # Compute the speed score and store in history
        speed_score = fuzzy.compute_speed(time_taken, difficulty)
        self._answer_history.append((correct, speed_score))

    def get_level(self) -> dict[str, Any]:
        """
        Run the fuzzy engine on the last WINDOW_SIZE answers and
        return the current performance level as a dictionary.
        """
        history = self._answer_history[-WINDOW_SIZE:]

        if not history:
            # No answers yet — return a neutral starting level
            return {
                "score":     5.0,
                "label":     "moderate",
                "accuracy":  0.5,
                "speed":     0.5,
                "difficulty": FIRST_DIFFICULTY,
            }

        # Calculate average accuracy and speed over the recent window
        accuracy = sum(1 for correct, _ in history if correct) / len(history)
        speed    = sum(spd for _, spd in history) / len(history)

        score = fuzzy.compute_level(accuracy, speed)
        return {
            "score":      round(score, 2),
            "label":      fuzzy.level_label(score),
            "accuracy":   round(accuracy, 2),
            "speed":      round(speed, 2),
            "difficulty": fuzzy.level_to_difficulty(score),
        }

    def next_question(self) -> dict | None:
        """
        Pick the next question for the student.

        Rule:
          1. Use the fuzzy level to choose a target difficulty.
          2. Choose the weakest topic 70% of the time, otherwise a random topic.
          3. Pick an unseen question matching that topic and difficulty.
          4. If none left, try the nearest difficulty; if still none, any unseen question.
          5. Return None if all questions have been asked.
        """
        unseen = self._all_questions[
            ~self._all_questions["id"].isin(self._asked_ids)
        ]
        if unseen.empty:
            return None

        # Decide target difficulty
        if not self._answer_history:
            target_difficulty = FIRST_DIFFICULTY
        else:
            level_info        = self.get_level()
            target_difficulty = level_info["difficulty"]

        # Decide target topic
        if self.target_topic and self.target_topic in ALL_TOPICS:
            target_topic = self.target_topic
        else:
            weak = self._mastery.weak_topics()
            if weak and random.random() < WEAK_TOPIC_CHANCE:
                target_topic = weak[0]   # weakest topic first
            else:
                target_topic = random.choice(ALL_TOPICS)

        # Try to find a question matching both topic and difficulty
        candidate = unseen[
            (unseen["topic"] == target_topic) &
            (unseen["difficulty"] == target_difficulty)
        ]

        # Fall back to the nearest difficulty if needed
        if candidate.empty:
            difficulty_order = ["Easy", "Medium", "Hard"]
            for diff in difficulty_order:
                candidate = unseen[
                    (unseen["topic"] == target_topic) &
                    (unseen["difficulty"] == diff)
                ]
                if not candidate.empty:
                    break

        # Fall back to any unseen question
        if candidate.empty:
            candidate = unseen

        # Pick one at random from the candidates
        chosen = candidate.sample(1, random_state=None).iloc[0]
        self._asked_ids.append(int(chosen["id"]))

        return {
            "id":         int(chosen["id"]),
            "question":   str(chosen["question"]),
            "option_a":   str(chosen["option_a"]),
            "option_b":   str(chosen["option_b"]),
            "option_c":   str(chosen["option_c"]),
            "option_d":   str(chosen["option_d"]),
            "answer":     str(chosen["answer"]),
            "topic":      str(chosen["topic"]),
            "difficulty": str(chosen["difficulty"]),
        }

    def topic_mastery(self) -> dict[str, float]:
        """Return a dictionary of topic -> mastery probability for all topics."""
        result = {}
        for topic in ALL_TOPICS:
            result[topic] = round(self._mastery.get(topic), 3)
        return result

    def weak_topics(self) -> list[str]:
        """Return the list of topics where mastery is below 0.6, weakest first."""
        return self._mastery.weak_topics()

    def recommendations(self) -> dict[str, str]:
        """
        For each weak topic, produce a short plain-English recommendation.
        Returns a dictionary of topic -> advice string.
        """
        advice: dict[str, str] = {}
        for topic in self.weak_topics():
            prob  = self._mastery.get(topic)
            level = self.get_level()["label"]

            if level == "weak" or prob < 0.4:
                difficulty_hint = "Easy and Medium"
            elif level == "strong":
                difficulty_hint = "Medium and Hard"
            else:
                difficulty_hint = "Easy, Medium and Hard"

            advice[topic] = (
                f"Practise more {difficulty_hint} questions on {topic}. "
                f"Your current mastery is {prob:.0%}."
            )
        return advice

    def summary(self) -> dict[str, Any]:
        """
        Build the summary dictionary shown on the Results page.

        Includes: score, accuracy, average time, topic mastery,
        mastery history, weak topics and recommendations.
        """
        history    = self._answer_history
        total      = len(history)
        n_correct  = sum(1 for correct, _ in history if correct)
        accuracy   = n_correct / total if total else 0.0

        # Pull detailed records from the database for time information
        attempts = db.get_attempts(self.student_id)
        # Only count attempts from this session (last `total` rows)
        session_attempts = attempts[-total:] if total else []
        avg_time = (
            sum(a["time_taken"] for a in session_attempts) / len(session_attempts)
            if session_attempts else 0.0
        )

        # Build mastery over time: after each answer, what was the mastery?
        mastery_history: list[dict] = []
        tracker = mastery_mod.MasteryTracker()
        for i, attempt in enumerate(session_attempts):
            tracker.update(attempt["topic"], bool(attempt["correct"]))
            mastery_history.append({
                "question": i + 1,
                **tracker.all(),
            })

        return {
            "total":           total,
            "correct":         n_correct,
            "accuracy":        round(accuracy, 3),
            "avg_time":        round(avg_time, 1),
            "topic_mastery":   self.topic_mastery(),
            "mastery_history": mastery_history,
            "weak_topics":     self.weak_topics(),
            "recommendations": self.recommendations(),
            "level":           self.get_level(),
        }
