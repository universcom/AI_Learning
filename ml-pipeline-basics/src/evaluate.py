from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classification_model(model, X_train, X_test, y_train, y_test) -> dict:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_prob = model.predict_proba(X_train)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    results = {
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "train_f1": f1_score(y_train, train_pred),
        "test_f1": f1_score(y_test, test_pred),
        "test_precision": precision_score(y_test, test_pred),
        "test_recall": recall_score(y_test, test_pred),
        "train_roc_auc": roc_auc_score(y_train, train_prob),
        "test_roc_auc": roc_auc_score(y_test, test_prob),
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
        "classification_report": classification_report(y_test, test_pred),
    }
    return results