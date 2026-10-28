"""
Simulate three types of students taking the quiz and save results.

Student types:
  - Strong:  ~85% correct answers, fast responses
  - Average: ~55% correct answers, average speed
  - Weak:    ~25% correct answers, slow responses

Run this from the project root:
    python scripts/simulate_students.py
"""

from __future__ import annotations

import pathlib
import random
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from adaptive_engine import db
from adaptive_engine.engine import AdaptiveEngine, ALL_TOPICS

# Fixed random seed for reproducibility
random.seed(99)

DOCS_DIR       = pathlib.Path("docs")
QUESTIONS_CSV  = pathlib.Path("data") / "questions.csv"
NUM_QUESTIONS  = 20   # questions per simulated student

# Define the three student types
STUDENT_TYPES = {
    "Strong":  {"correct_rate": 0.85, "time_mean": 15.0, "time_std": 5.0},
    "Average": {"correct_rate": 0.55, "time_mean": 40.0, "time_std": 10.0},
    "Weak":    {"correct_rate": 0.25, "time_mean": 70.0, "time_std": 15.0},
}

# Use simulation-only student IDs that won't clash with real accounts
SIM_STUDENT_IDS = {"Strong": 201, "Average": 202, "Weak": 203}


def simulate_one_student(
    student_type: str,
    correct_rate: float,
    time_mean: float,
    time_std: float,
) -> pd.DataFrame:
    """
    Simulate one student answering NUM_QUESTIONS questions.

    Returns a DataFrame recording each question's topic, difficulty,
    and whether the answer was correct.
    """
    student_id = SIM_STUDENT_IDS[student_type]

    # Insert a temporary student row so the engine can save attempts
    conn = db.get_connection()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO students (student_id, name, password) VALUES (?, ?, ?)",
            (student_id, f"Sim {student_type}", "x"),
        )
        # Clear any previous simulation attempts for this ID
        conn.execute("DELETE FROM attempts WHERE student_id = ?", (student_id,))
    conn.close()

    engine = AdaptiveEngine(student_id=student_id)
    records: list[dict] = []

    for _ in range(NUM_QUESTIONS):
        question = engine.next_question()
        if question is None:
            break

        # Decide if this simulated student gets it right
        correct    = random.random() < correct_rate
        time_taken = max(5.0, random.gauss(time_mean, time_std))

        engine.record_answer(question["id"], correct, time_taken)

        records.append({
            "type":       student_type,
            "topic":      question["topic"],
            "difficulty": question["difficulty"],
            "correct":    int(correct),
            "time_taken": round(time_taken, 1),
        })

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame) -> None:
    """Print a table showing difficulty distribution per student type."""
    print("\n=== Simulation Summary ===\n")
    pivot = (
        df.groupby(["type", "difficulty"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Easy", "Medium", "Hard"], fill_value=0)
    )
    print(pivot.to_string())
    print()

    for stype in ["Strong", "Average", "Weak"]:
        subset   = df[df["type"] == stype]
        accuracy = subset["correct"].mean()
        print(f"{stype:8s}  accuracy={accuracy:.0%}  "
              f"avg_time={subset['time_taken'].mean():.1f}s  "
              f"questions={len(subset)}")


def save_results(df: pd.DataFrame) -> None:
    """Save the simulation DataFrame to CSV and a bar chart to PNG."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw data
    csv_path = DOCS_DIR / "simulation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Build a difficulty-count chart
    pivot = (
        df.groupby(["type", "difficulty"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Easy", "Medium", "Hard"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(
        kind   = "bar",
        ax     = ax,
        color  = ["#10b981", "#f59e0b", "#ef4444"],
        edgecolor = "white",
        width  = 0.6,
    )
    ax.set_title(
        "Question Difficulty Distribution by Student Type",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Student Type", fontsize=11)
    ax.set_ylabel("Number of Questions", fontsize=11)
    ax.set_xticklabels(pivot.index, rotation=0, fontsize=11)
    ax.legend(title="Difficulty", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    png_path = DOCS_DIR / "simulation.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Saved {png_path}")


def main() -> None:
    """Run all three student simulations and save results."""
    db.initialise_db()

    all_results: list[pd.DataFrame] = []
    for stype, params in STUDENT_TYPES.items():
        print(f"Simulating {stype} student …")
        result_df = simulate_one_student(stype, **params)
        all_results.append(result_df)

    combined = pd.concat(all_results, ignore_index=True)
    print_summary(combined)
    save_results(combined)


if __name__ == "__main__":
    main()
