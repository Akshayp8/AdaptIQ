"""
Train the NLP topic classifier and save the model and a confusion matrix plot.

Run this from the project root:
    python scripts/train_classifier.py
"""

from __future__ import annotations

import pathlib
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Make sure the src/ folder is on the path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

QUESTIONS_CSV = pathlib.Path("data") / "questions.csv"
MODEL_PATH    = pathlib.Path("models") / "topic_classifier.joblib"
DOCS_DIR      = pathlib.Path("docs")


def load_data() -> tuple[list[str], list[str]]:
    """Load the question bank and build a combined text column plus labels."""
    df = pd.read_csv(QUESTIONS_CSV)
    texts = (
        df["question"].fillna("").astype(str) + " " +
        df["option_a"].fillna("").astype(str) + " " +
        df["option_b"].fillna("").astype(str) + " " +
        df["option_c"].fillna("").astype(str) + " " +
        df["option_d"].fillna("").astype(str)
    ).tolist()
    labels = df["topic"].tolist()
    return texts, labels


def build_pipeline() -> Pipeline:
    """Create and return a TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42)),
    ])


def save_confusion_matrix(y_true: list, y_pred: list, labels: list[str]) -> None:
    """Draw the confusion matrix and save it to docs/confusion_matrix.png."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Topic Classifier — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    out_path = DOCS_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {out_path}")


def main() -> None:
    """Train the classifier, print the report, save the model and confusion matrix."""
    print("Loading data …")
    texts, labels = load_data()
    unique_labels = sorted(set(labels))

    # 75% train, 25% test split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")

    print("Training …")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = sum(p == t for p, t in zip(y_pred, y_test)) / len(y_test)
    print(f"\nTest accuracy: {accuracy:.2%}\n")
    print(classification_report(y_test, y_pred))

    save_confusion_matrix(y_test, y_pred, unique_labels)

    # Save the trained model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
