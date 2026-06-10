"""
model_evaluation.py
-------------------
Evaluation metrics, comparison tables, and diagnostic plots.

All metrics are computed in log-price space.
Metric keys use plain ASCII names ('R2', 'MAE', 'MSE', 'RMSE') for
compatibility with the rest of the pipeline.
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, model_name: str = "") -> dict:
    """
    Return a metrics dict with plain ASCII keys:
        Model, R2, MAE, MSE, RMSE
    All values are computed in log-price space.
    """
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    metrics = {"Model": model_name, "R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}
    logger.info(
        "%-30s  R2=%.4f  MAE=%.4f  RMSE=%.4f",
        model_name, r2, mae, rmse,
    )
    return metrics


def evaluate_all_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate every trained model and return a DataFrame sorted by R2 (descending).
    Columns: Model, R2, MAE, MSE, RMSE
    """
    records = []
    for name, pipeline in trained_models.items():
        y_pred = pipeline.predict(X_test)
        records.append(compute_metrics(y_test, y_pred, model_name=name))
    results_df = (
        pd.DataFrame(records)
        .sort_values("R2", ascending=False)
        .reset_index(drop=True)
    )
    return results_df


def plot_actual_vs_predicted(
    y_test, y_pred, model_name: str, save_path=None
) -> None:
    """Scatter plot: actual vs predicted values in log-price space."""
    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test_arr, y_pred_arr, alpha=0.5, edgecolors="none",
               color="steelblue", s=40)
    lim = [
        min(y_test_arr.min(), y_pred_arr.min()) - 0.1,
        max(y_test_arr.max(), y_pred_arr.max()) + 0.1,
    ]
    ax.plot(lim, lim, "r--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual log(Selling Price)", fontsize=12)
    ax.set_ylabel("Predicted log(Selling Price)", fontsize=12)
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=14,
                 fontweight="bold")
    ax.legend()
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=150)
        logger.info("Saved: %s", save_path)
    plt.close(fig)


def plot_residuals(
    y_test, y_pred, model_name: str, save_path=None
) -> None:
    """Two-panel residual diagnostic: residuals vs predicted + error histogram."""
    residuals = np.asarray(y_test) - np.asarray(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.5, color="darkorange",
                    edgecolors="none", s=40)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Value", fontsize=12)
    axes[0].set_ylabel("Residual", fontsize=12)
    axes[0].set_title(
        f"Residuals vs Predicted — {model_name}", fontsize=13, fontweight="bold"
    )

    axes[1].hist(residuals, bins=30, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Error Distribution", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=150)
        logger.info("Saved: %s", save_path)
    plt.close(fig)


def plot_model_comparison(results_df: pd.DataFrame, save_path=None) -> None:
    """Horizontal bar chart comparing R2 scores across all models."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(results_df)))
    bars = ax.barh(
        results_df["Model"], results_df["R2"],
        color=colors[::-1], edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
    ax.set_xlabel("R² Score", fontsize=12)
    ax.set_title(
        "Model Comparison — R² Score (Test Set)", fontsize=14, fontweight="bold"
    )
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=150)
        logger.info("Saved: %s", save_path)
    plt.close(fig)


def plot_feature_importance(
    model_pipeline, feature_names: list, model_name: str, save_path=None
) -> None:
    """
    Horizontal bar chart of feature importances (tree-based models only).
    Silently skips linear models that lack feature_importances_.
    """
    model = model_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        logger.warning(
            "%s does not expose feature_importances_ — skipping importance plot.",
            model_name,
        )
        return

    importances = model.feature_importances_
    fi_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=True)
        .tail(15)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(fi_df["Feature"], fi_df["Importance"],
            color="steelblue", edgecolor="white")
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(
        f"Top Feature Importances — {model_name}", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), dpi=150)
        logger.info("Saved: %s", save_path)
    plt.close(fig)


if __name__ == "__main__":
    print("model_evaluation.py loaded successfully.")
