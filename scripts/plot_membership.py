"""
Plot all fuzzy membership functions and save them to the docs/ folder.

Run this from the project root:
    python scripts/plot_membership.py
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

DOCS_DIR = pathlib.Path("docs")


def triangular(universe: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Compute a triangular membership function over a universe array.
    a = left edge (membership 0), b = peak (membership 1), c = right edge (membership 0).
    """
    left  = np.maximum(0, (universe - a) / (b - a + 1e-9))
    right = np.maximum(0, (c - universe) / (c - b + 1e-9))
    return np.minimum(left, right)


def plot_variable(
    universe: np.ndarray,
    terms: dict[str, tuple[float, float, float]],
    title: str,
    xlabel: str,
    filename: str,
) -> None:
    """Draw all membership functions for one fuzzy variable and save the plot."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    colors = ["#6366f1", "#0ea5e9", "#10b981"]
    fig, ax = plt.subplots(figsize=(8, 4))

    for (label, (a, b, c)), color in zip(terms.items(), colors):
        y = triangular(universe, a, b, c)
        ax.plot(universe, y, label=label, color=color, linewidth=2.5)
        ax.fill_between(universe, y, alpha=0.12, color=color)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Membership degree", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = DOCS_DIR / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    """Generate and save all three membership function plots."""
    # Accuracy (0 to 1)
    acc_universe = np.linspace(0, 1, 200)
    plot_variable(
        universe = acc_universe,
        terms    = {
            "low":    (0.0, 0.0, 0.5),
            "medium": (0.2, 0.5, 0.8),
            "high":   (0.5, 1.0, 1.0),
        },
        title    = "Accuracy — Fuzzy Membership Functions",
        xlabel   = "Accuracy (fraction correct)",
        filename = "membership_accuracy.png",
    )

    # Speed (0 to 1)
    spd_universe = np.linspace(0, 1, 200)
    plot_variable(
        universe = spd_universe,
        terms    = {
            "slow":    (0.0, 0.0, 0.5),
            "average": (0.2, 0.5, 0.8),
            "fast":    (0.5, 1.0, 1.0),
        },
        title    = "Speed — Fuzzy Membership Functions",
        xlabel   = "Speed score (1 = fast, 0 = slow)",
        filename = "membership_speed.png",
    )

    # Level (0 to 10)
    lvl_universe = np.linspace(0, 10, 200)
    plot_variable(
        universe = lvl_universe,
        terms    = {
            "weak":     (0.0,  0.0, 5.0),
            "moderate": (2.5,  5.0, 7.5),
            "strong":   (5.0, 10.0, 10.0),
        },
        title    = "Level — Fuzzy Membership Functions",
        xlabel   = "Level score (0–10)",
        filename = "membership_level.png",
    )

    print("\nAll membership plots saved to docs/")


if __name__ == "__main__":
    main()
