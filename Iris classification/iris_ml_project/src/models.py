"""
models.py
---------
Model registry and training pipeline.
Add or remove classifiers by editing MODEL_REGISTRY — no other file needs changing.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE, CV_FOLDS


# ── Registry ──────────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE
    ),
    "Support Vector Machine": SVC(
        kernel="rbf", C=1.0, gamma="scale",
        probability=True, random_state=RANDOM_STATE,
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=200, random_state=RANDOM_STATE
    ),
}


# ── Training ──────────────────────────────────────────────────────────────────

def train_all(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    le,
    scaler:  StandardScaler,
    X_full:  np.ndarray,
    y_full:  np.ndarray,
) -> dict:
    """
    Fit every model in MODEL_REGISTRY and collect evaluation metrics.

    Parameters
    ----------
    X_train / X_test  : scaled feature arrays (from data_loader.build_splits)
    y_train / y_test  : integer label arrays
    le                : fitted LabelEncoder (for class names)
    scaler            : fitted StandardScaler (for cross-val on full data)
    X_full / y_full   : unscaled full dataset (scaled internally for CV)

    Returns
    -------
    results : dict[model_name -> metric_dict]
    """
    results: dict = {}

    for name, model in MODEL_REGISTRY.items():
        # ── Fit ───────────────────────────────────────────────────────────────
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ── Metrics ───────────────────────────────────────────────────────────
        acc    = accuracy_score(y_test, y_pred)
        cv     = cross_val_score(
            model, scaler.transform(X_full), y_full, cv=CV_FOLDS, scoring="accuracy"
        )
        report = classification_report(
            y_test, y_pred, target_names=le.classes_, output_dict=True
        )
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model":  model,
            "acc":    acc,
            "cv":     cv,
            "report": report,
            "cm":     cm,
            "y_pred": y_pred,
        }

        print(
            f"  ✓ {name:<28s}  "
            f"acc={acc*100:.1f}%  "
            f"cv={cv.mean()*100:.1f}% ± {cv.std()*100:.1f}%"
        )

    return results


def get_best(results: dict) -> tuple[str, dict]:
    """Return (name, metrics) for the highest test-accuracy model."""
    best_name = max(results, key=lambda k: results[k]["acc"])
    return best_name, results[best_name]


def get_feature_importances(results: dict, feature_names: list) -> dict:
    """
    Extract feature importances from tree-based models.
    Returns {feature_name: importance} sorted descending.
    """
    for preferred in ("Random Forest", "Decision Tree"):
        if preferred in results:
            model = results[preferred]["model"]
            imp   = model.feature_importances_
            return dict(
                sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)
            )
    return {}
