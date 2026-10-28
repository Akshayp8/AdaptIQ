"""
Fuzzy logic module for the Adaptive AI Learning System.

Reads the student's recent accuracy and speed, and uses fuzzy rules
to decide how strong the student currently is (weak / moderate / strong).
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ---------------------------------------------------------------------------
# Build the fuzzy system once when this module is imported.
# Re-using one system is faster than rebuilding it for every answer.
# ---------------------------------------------------------------------------

# --- Input variables --------------------------------------------------------

# accuracy: fraction of recent answers that were correct (0 = none, 1 = all)
_accuracy = ctrl.Antecedent(np.linspace(0, 1, 101), "accuracy")

# speed: how fast the student answered compared to expected time (0 = very slow, 1 = very fast)
_speed = ctrl.Antecedent(np.linspace(0, 1, 101), "speed")

# --- Output variable --------------------------------------------------------

# level: overall performance score from 0 (worst) to 10 (best)
_level = ctrl.Consequent(np.linspace(0, 10, 101), "level")

# --- Triangular membership functions ----------------------------------------
# Each function is defined by three points: (left edge, peak, right edge).

# accuracy terms
_accuracy["low"]    = fuzz.trimf(_accuracy.universe, [0.0, 0.0, 0.5])
_accuracy["medium"] = fuzz.trimf(_accuracy.universe, [0.2, 0.5, 0.8])
_accuracy["high"]   = fuzz.trimf(_accuracy.universe, [0.5, 1.0, 1.0])

# speed terms
_speed["slow"]    = fuzz.trimf(_speed.universe, [0.0, 0.0, 0.5])
_speed["average"] = fuzz.trimf(_speed.universe, [0.2, 0.5, 0.8])
_speed["fast"]    = fuzz.trimf(_speed.universe, [0.5, 1.0, 1.0])

# level terms
_level["weak"]     = fuzz.trimf(_level.universe, [0.0, 0.0, 5.0])
_level["moderate"] = fuzz.trimf(_level.universe, [2.5, 5.0, 7.5])
_level["strong"]   = fuzz.trimf(_level.universe, [5.0, 10.0, 10.0])

# --- Exactly 9 rules (every combination of accuracy × speed) ----------------
# Rule format: accuracy X AND speed Y  ->  level Z
_rules = [
    ctrl.Rule(_accuracy["high"]   & _speed["fast"],    _level["strong"]),   # 1
    ctrl.Rule(_accuracy["high"]   & _speed["average"], _level["strong"]),   # 2
    ctrl.Rule(_accuracy["high"]   & _speed["slow"],    _level["moderate"]), # 3
    ctrl.Rule(_accuracy["medium"] & _speed["fast"],    _level["moderate"]), # 4
    ctrl.Rule(_accuracy["medium"] & _speed["average"], _level["moderate"]), # 5
    ctrl.Rule(_accuracy["medium"] & _speed["slow"],    _level["weak"]),     # 6
    ctrl.Rule(_accuracy["low"]    & _speed["fast"],    _level["moderate"]), # 7
    ctrl.Rule(_accuracy["low"]    & _speed["average"], _level["weak"]),     # 8
    ctrl.Rule(_accuracy["low"]    & _speed["slow"],    _level["weak"]),     # 9
]

# Build the control system and one simulation object
_control_system = ctrl.ControlSystem(_rules)
_simulation     = ctrl.ControlSystemSimulation(_control_system)

# Expected answer times in seconds per difficulty level
EXPECTED_TIME = {"Easy": 20.0, "Medium": 40.0, "Hard": 60.0}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_speed(time_taken: float, difficulty: str) -> float:
    """
    Turn the raw time taken (seconds) into a speed score between 0 and 1.

    Speed = 1 means the student answered instantly (much faster than expected).
    Speed = 0 means the student took a very long time.

    Example: expected time for Medium is 40 s.
      - took 20 s  -> speed = min(1, 40/20) = 1.0  (fast)
      - took 40 s  -> speed = min(1, 40/40) = 1.0  (right on time)
      - took 80 s  -> speed = min(1, 40/80) = 0.5  (slow)
      - took 160 s -> speed = min(1, 40/160) = 0.25 (very slow)
    """
    expected = EXPECTED_TIME.get(difficulty, 40.0)
    if time_taken <= 0:
        return 1.0
    raw_speed = expected / time_taken
    return float(min(1.0, raw_speed))


def compute_level(accuracy: float, speed: float) -> float:
    """
    Run the fuzzy inference engine and return a level score from 0 to 10.

    accuracy -- fraction of recent answers correct (0.0 to 1.0)
    speed    -- average speed score of recent answers (0.0 to 1.0)

    If the fuzzy engine cannot compute a result (e.g. no rules fire),
    we fall back to a simple average of the two inputs scaled to 10.
    """
    # Clamp inputs to valid range so the engine never crashes
    acc = float(np.clip(accuracy, 0.01, 0.99))
    spd = float(np.clip(speed,    0.01, 0.99))

    try:
        _simulation.input["accuracy"] = acc
        _simulation.input["speed"]    = spd
        _simulation.compute()
        return float(_simulation.output["level"])
    except Exception:
        # Fallback: simple weighted average
        return float((acc * 0.7 + spd * 0.3) * 10)


def level_label(level: float) -> str:
    """
    Convert a numeric level (0 to 10) into a human-readable label.

    0–4   -> "weak"
    4–6   -> "moderate"
    6–10  -> "strong"
    """
    if level < 4.0:
        return "weak"
    if level < 6.5:
        return "moderate"
    return "strong"


def level_to_difficulty(level: float) -> str:
    """
    Convert a numeric level (0 to 10) into a question difficulty.

    0–3.5  -> "Easy"
    3.5–6.5 -> "Medium"
    6.5–10  -> "Hard"
    """
    if level < 3.5:
        return "Easy"
    if level < 6.5:
        return "Medium"
    return "Hard"
