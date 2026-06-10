"""
main.py
-------
Entry point for the Iris Flower Classification pipeline.

Usage
-----
    python src/main.py

Output
------
    outputs/iris_classification_report.html
"""

import os
import sys
import time

# Allow running from project root or from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FEATURES, OUTPUT_DIR
from data_loader import build_splits, get_dataset_summary, load_raw
from models import get_best, get_feature_importances, train_all
from report import build_report
from visualiser import (
    plot_boxplots,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_cv_comparison,
    plot_feature_distributions,
    plot_feature_importance,
    plot_model_comparison,
    plot_pairplot,
)


def main() -> None:
    t0 = time.time()
    sep = "-" * 58

    print("\n  Iris Flower Classification Pipeline")
    print(f"    {sep}\n")

    print("  [1/4] Loading and preprocessing data...")
    df = load_raw()
    summary = get_dataset_summary(df)
    X_train, X_test, y_train, y_test, le, scaler = build_splits(df)
    X_full = df[FEATURES].values
    y_full = le.transform(df["Species"])
    print(
        f"       {summary['n_samples']} samples, "
        f"{summary['n_features']} features, "
        f"{summary['n_classes']} classes "
        f"| missing: {summary['missing_values']}"
    )

    print("\n  [2/4] Training models...")
    results = train_all(
        X_train,
        X_test,
        y_train,
        y_test,
        le,
        scaler,
        X_full,
        y_full,
    )

    best_name, best = get_best(results)
    print(f"\n       Best model: {best_name} ({best['acc'] * 100:.1f}% test accuracy)")

    importances = get_feature_importances(results, FEATURES)

    print("\n  [3/4] Generating visualisations...")
    images = {
        "donut": plot_class_distribution(df),
        "distributions": plot_feature_distributions(df),
        "boxplots": plot_boxplots(df),
        "pairplot": plot_pairplot(df),
        "correlation": plot_correlation_heatmap(df),
        "comparison": plot_model_comparison(results, best_name),
        "cv": plot_cv_comparison(results),
        "confusion": plot_confusion_matrix(best["cm"], best_name),
        "importance": plot_feature_importance(importances),
    }
    print(f"       {len(images)} charts rendered")

    print("\n  [4/4] Building HTML report...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = build_report(
        results=results,
        best_name=best_name,
        summary=summary,
        images=images,
        importances=importances,
    )

    elapsed = time.time() - t0
    print(f"\n  {sep}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Report: {output_path}\n")


if __name__ == "__main__":
    main()
