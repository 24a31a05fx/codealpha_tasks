"""
main.py
-------
End-to-end entry point for the Car Price Prediction ML pipeline.
"""

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_preprocessing import clean_pipeline, inspect_data
from src.feature_engineering import feature_engineering_pipeline
from src.model_evaluation import (
    compute_metrics,
    evaluate_all_models,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_model_comparison,
    plot_residuals,
)
from src.model_training import prepare_features, split_data, train_all_models, tune_best_model

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "car_data.csv"
PLOTS_DIR = ROOT_DIR / "outputs" / "plots"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
MODELS_DIR = ROOT_DIR / "models"


def ensure_output_dirs() -> None:
    """Create output directories if they do not already exist."""
    for path in [PLOTS_DIR, REPORTS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def run_eda(df_clean: pd.DataFrame) -> None:
    """Generate exploratory data analysis plots."""
    logger.info("STEP 2: Exploratory data analysis")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_clean["selling_price"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("Selling Price Distribution", fontweight="bold")
    axes[0].set_xlabel("Selling Price (Lakhs INR)")
    axes[0].set_ylabel("Count")

    axes[1].hist(np.log1p(df_clean["selling_price"]), bins=30, color="darkorange", edgecolor="white")
    axes[1].set_title("Log Selling Price Distribution", fontweight="bold")
    axes[1].set_xlabel("log(Selling Price)")
    axes[1].set_ylabel("Count")
    plt.suptitle("Target Variable Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_target_distribution.png", dpi=150)
    plt.close(fig)

    num_cols = ["present_price", "driven_kms", "year"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, num_cols):
        ax.hist(df_clean[col].dropna(), bins=25, color="teal", edgecolor="white")
        ax.set_title(col.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    plt.suptitle("Numerical Feature Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_numerical_histograms.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        df_clean["present_price"],
        df_clean["selling_price"],
        alpha=0.5,
        color="royalblue",
        edgecolors="none",
        s=40,
    )
    ax.set_xlabel("Present Price (Lakhs INR)", fontsize=12)
    ax.set_ylabel("Selling Price (Lakhs INR)", fontsize=12)
    ax.set_title("Selling Price vs Present Price", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_price_scatter.png", dpi=150)
    plt.close(fig)

    cat_plots = [
        ("fuel_type", "Selling Price by Fuel Type"),
        ("transmission", "Selling Price by Transmission"),
        ("selling_type", "Selling Price by Seller Type"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, title) in zip(axes, cat_plots):
        order = df_clean.groupby(col)["selling_price"].median().sort_values(ascending=False).index
        sns.boxplot(data=df_clean, x=col, y="selling_price", order=order, ax=ax, palette="Set2")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Selling Price (Lakhs)")
    plt.suptitle("Categorical Feature Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_categorical_boxplots.png", dpi=150)
    plt.close(fig)

    num_df = df_clean.select_dtypes(include=np.number)
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    logger.info("EDA plots saved to %s", PLOTS_DIR)


def run_pipeline() -> None:
    """Run cleaning, EDA, feature engineering, training, evaluation, and saving."""
    ensure_output_dirs()
    sns.set_theme(style="whitegrid", palette="muted")

    logger.info("STEP 1: Loading and cleaning data")
    df_clean = clean_pipeline(str(DATA_PATH))
    inspect_data(df_clean)

    run_eda(df_clean)

    logger.info("STEP 3: Feature engineering")
    df_engineered = feature_engineering_pipeline(df_clean.copy())

    logger.info("STEP 4: Preparing features and splitting data")
    X, y, feature_names = prepare_features(df_engineered)
    X_train, X_test, y_train, y_test = split_data(X, y)

    logger.info("STEP 5: Training models")
    trained_models = train_all_models(X_train, y_train)

    logger.info("STEP 6: Evaluating models")
    results_df = evaluate_all_models(trained_models, X_test, y_test)
    print("\n--- Model Comparison ---")
    print(results_df.to_string(index=False))
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    plot_model_comparison(results_df, save_path=REPORTS_DIR.parent / "plots" / "06_model_comparison.png")

    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_models[best_model_name]
    y_pred_best = best_pipeline.predict(X_test)

    plot_actual_vs_predicted(
        y_test,
        y_pred_best,
        best_model_name,
        save_path=PLOTS_DIR / "07_actual_vs_predicted.png",
    )
    plot_residuals(
        y_test,
        y_pred_best,
        best_model_name,
        save_path=PLOTS_DIR / "08_residuals.png",
    )
    plot_feature_importance(
        best_pipeline,
        feature_names,
        best_model_name,
        save_path=PLOTS_DIR / "09_feature_importance.png",
    )

    logger.info("STEP 7: Hyperparameter tuning (Random Forest)")
    tuned_pipeline = tune_best_model(X_train, y_train)
    y_pred_tuned = tuned_pipeline.predict(X_test)
    tuned_metrics = compute_metrics(y_test, y_pred_tuned, "Random Forest (Tuned)")
    print(f"\nTuned model -> R2={tuned_metrics['R2']:.4f}  RMSE={tuned_metrics['RMSE']:.4f}")

    plot_actual_vs_predicted(
        y_test,
        y_pred_tuned,
        "Random Forest (Tuned)",
        save_path=PLOTS_DIR / "10_tuned_actual_vs_predicted.png",
    )
    plot_feature_importance(
        tuned_pipeline,
        feature_names,
        "Random Forest (Tuned)",
        save_path=PLOTS_DIR / "11_tuned_feature_importance.png",
    )

    logger.info("STEP 8: Saving model artifacts")
    if tuned_metrics["R2"] > results_df.iloc[0]["R2"]:
        final_model_name = "Random Forest (Tuned)"
        final_pipeline = tuned_pipeline
        final_r2 = tuned_metrics["R2"]
        final_rmse = tuned_metrics["RMSE"]
    else:
        final_model_name = best_model_name
        final_pipeline = best_pipeline
        final_r2 = results_df.iloc[0]["R2"]
        final_rmse = results_df.iloc[0]["RMSE"]

    model_path = MODELS_DIR / "best_model.pkl"
    tuned_model_path = MODELS_DIR / "tuned_random_forest.pkl"
    feature_path = MODELS_DIR / "feature_names.pkl"
    joblib.dump(final_pipeline, model_path)
    joblib.dump(tuned_pipeline, tuned_model_path)
    joblib.dump(feature_names, feature_path)

    print("\n" + "=" * 60)
    print("  CAR PRICE PREDICTION - PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Saved best model : {final_model_name}")
    print(f"  Best R2          : {final_r2:.4f}")
    print(f"  Best RMSE        : {final_rmse:.4f}")
    print(f"  Saved model     : {model_path}")
    print(f"  Tuned RF saved  : {tuned_model_path}")
    print(f"  Plots saved to  : {PLOTS_DIR}")
    print(f"  Report saved to : {REPORTS_DIR / 'model_comparison.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
