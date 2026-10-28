"""
NLP Topic Classifier for the Adaptive AI Learning System.

Uses TF-IDF to turn question text into numbers, then a Logistic Regression
model to predict which topic the question belongs to.
This is useful when someone adds a new question without picking a topic.
"""

from __future__ import annotations

import pathlib

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Path to the saved model file
MODEL_PATH = pathlib.Path("models") / "topic_classifier.joblib"

# A module-level variable that holds the loaded model (loaded lazily)
_model: Pipeline | None = None


def _train_and_save(questions_csv: pathlib.Path) -> Pipeline:
    """
    Train a TF-IDF + Logistic Regression classifier on the question bank,
    save it to disk, and return the trained pipeline.
    """
    df = pd.read_csv(questions_csv)

    # Build a simple text column: combine the question and all four options safely as strings
    X = (
        df["question"].fillna("").astype(str) + " " +
        df["option_a"].fillna("").astype(str) + " " +
        df["option_b"].fillna("").astype(str) + " " +
        df["option_c"].fillna("").astype(str) + " " +
        df["option_d"].fillna("").astype(str)
    )
    y = df["topic"]

    # Build the pipeline: vectorise text, then classify
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X, y)

    # Save the model so we do not need to retrain every time
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def load_model(questions_csv: pathlib.Path | None = None) -> Pipeline:
    """
    Return the topic classifier model.

    If a saved model exists on disk, load it.
    Otherwise, train a new one from the question CSV and save it.
    """
    global _model
    if _model is not None:
        return _model

    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    else:
        csv_path = questions_csv or pathlib.Path("data") / "questions.csv"
        _model = _train_and_save(csv_path)

    return _model


def predict_topic(question_text: str,
                  questions_csv: pathlib.Path | None = None) -> str:
    """
    Predict the topic of a new question from its text.

    Returns a string like "Functions" or "Loops and Conditionals".
    """
    model = load_model(questions_csv)
    prediction = model.predict([question_text])
    return str(prediction[0])
