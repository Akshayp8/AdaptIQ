"""
Bayesian mastery tracking module.

For each topic we keep one number: the probability that the student
has truly mastered that topic.  We update it after every answer
using Bayes' rule, then add a small learning boost.
"""

from __future__ import annotations

# Fixed constants for the Bayesian update
PRIOR          = 0.30   # Starting probability of mastery for any topic
GUESS_RATE     = 0.20   # Chance of getting the answer right by guessing (not knowing)
SLIP_RATE      = 0.10   # Chance of getting the answer wrong even when you know it
LEARNING_BOOST = 0.15   # Small extra gain added after every correct answer
MAX_MASTERY    = 0.95   # Upper bound preventing Bayesian absorbing boundary (Cromwell's rule)
MIN_MASTERY    = 0.01   # Lower bound ensuring recovery from repeated slips


class MasteryTracker:
    """
    Tracks the probability that a student has mastered each topic.

    Plain-English explanation of the Bayesian update:
    -------------------------------------------------
    We start with a prior: P(knows) = 0.30.

    If the student answers CORRECTLY, we ask:
      "How likely is a correct answer from someone who knows?" = 1 - slip = 0.90
      "How likely is a correct answer from someone who doesn't know?" = guess = 0.20

    Bayes' formula (correct answer):
      P(knows | correct) = P(knows) * 0.90
                           / (P(knows) * 0.90 + (1 - P(knows)) * 0.20)

    If the student answers INCORRECTLY, we ask the opposite:
      P(knows | wrong) = P(knows) * slip
                         / (P(knows) * slip + (1 - P(knows)) * (1 - guess))

    After a correct answer we also add a learning boost of 0.15,
    because every attempt teaches the student something new.

    Worked example (correct answer, prior = 0.30):
      numerator   = 0.30 * 0.90 = 0.27
      denominator = 0.27 + 0.70 * 0.20 = 0.27 + 0.14 = 0.41
      posterior   = 0.27 / 0.41 ≈ 0.659
      after boost = min(1.0, 0.659 + 0.15) ≈ 0.809
    """

    def __init__(self) -> None:
        # Dictionary mapping topic name -> mastery probability (0 to 1)
        self._mastery: dict[str, float] = {}

    def _get_or_create(self, topic: str) -> float:
        """Return the current mastery for a topic, creating it with PRIOR if unseen."""
        if topic not in self._mastery:
            self._mastery[topic] = PRIOR
        return self._mastery[topic]

    def update(self, topic: str, correct: bool) -> None:
        """
        Update the mastery probability for a topic after one answer.

        Uses Bayes' rule with guess and slip rates, then adds a
        learning boost if the answer was correct.
        """
        p = self._get_or_create(topic)

        if correct:
            # Numerator: student knows AND answered correctly
            numerator = p * (1.0 - SLIP_RATE)
            # Denominator: all ways to get a correct answer
            denominator = numerator + (1.0 - p) * GUESS_RATE
            posterior = numerator / denominator
            # Learning boost: answering correctly means they learned something
            posterior = min(MAX_MASTERY, posterior + LEARNING_BOOST)
        else:
            # Numerator: student knows BUT made a slip
            numerator = p * SLIP_RATE
            # Denominator: all ways to get a wrong answer
            denominator = numerator + (1.0 - p) * (1.0 - GUESS_RATE)
            posterior = max(MIN_MASTERY, numerator / denominator)

        self._mastery[topic] = float(posterior)

    def get(self, topic: str) -> float:
        """Return the mastery probability for a single topic."""
        return self._get_or_create(topic)

    def all(self) -> dict[str, float]:
        """Return a dictionary of all topic mastery probabilities."""
        return dict(self._mastery)

    def weak_topics(self, threshold: float = 0.6) -> list[str]:
        """
        Return a list of topics where mastery is below the threshold (default 0.6).
        The weakest topic (lowest mastery) comes first.
        """
        weak = [(topic, prob) for topic, prob in self._mastery.items()
                if prob < threshold]
        # Sort by probability ascending so the weakest topic is first
        weak.sort(key=lambda x: x[1])
        return [topic for topic, _ in weak]


# We need numpy for clipping — import it at the top of this private section
import numpy as np  # noqa: E402  (placed after class to keep the docstring near the top)
