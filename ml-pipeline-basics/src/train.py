from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_dataset
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor


MODEL_PATH = Path("models/model.joblib")


def prepare_dataset():
    df = load_dataset()

    selected_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["survived"]
    df = df[selected_columns].copy()

    X = df.drop(columns=["survived"])
    y = df["survived"].astype(int)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def build_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()
    model = LogisticRegression(max_iter=1000, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def main() -> None:
    X_train, X_test, y_train, y_test = prepare_dataset()

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("F1 Score:", round(f1_score(y_test, y_pred), 4))
    print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()