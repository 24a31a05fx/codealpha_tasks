"""
main.py
-------
Entry point for the Iris Flower Classification pipeline.

Usage
-----
    cd src
    python main.py

Output
------
    ../outputs/iris_classification_report.html
"""

import sys
import time
import os

# Allow running from project root or from src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, FEATURES
from data_loader import load_raw, get_dataset_summary, build_splits
from models import train_all, get_best, get_feature_importances
from visualiser import (
    plot_class_distribution,
    plot_feature_distributions,
    plot_boxplots,
    plot_pairplot,
    plot_correlation_heatmap,
    plot_model_comparison,
    plot_cv_comparison,
    plot_confusion_matrix,
    plot_feature_importance,
)
from report import build_report


def main() -> None:
    t0 = time.time()
    sep = "─" * 58

    print(f"\n{'':>2}🌸  Iris Flower Classification Pipeline")
    print(f"{'':>4}{sep}\n")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("  [1/4] Loading & preprocessing data …")
    df      = load_raw()
    summary = get_dataset_summary(df)
    X_train, X_test, y_train, y_test, le, scaler = build_splits(df)
    X_full = df[FEATURES].values
    y_full = le.transform(df["Species"])
    print(f"       {summary['n_samples']} samples, "
          f"{summary['n_features']} features, "
          f"{summary['n_classes']} classes  "
          f"| missing: {summary['missing_values']}")

    # ── 2. Train ──────────────────────────────────────────────────────────────
    print(f"\n  [2/4] Training models …")
    results = train_all(X_train, X_test, y_train, y_test, le, scaler,
                        X_full, y_full)

    best_name, best = get_best(results)
    print(f"\n       🏆  Best model: {best_name}  "
          f"({best['acc']*100:.1f}% test accuracy)")

    importances = get_feature_importances(results, FEATURES)

    # ── 3. Visualise ──────────────────────────────────────────────────────────
    print(f"\n  [3/4] Generating visualisations …")
    images = {
        "donut":         plot_class_distribution(df),
        "distributions": plot_feature_distributions(df),
        "boxplots":      plot_boxplots(df),
        "pairplot":      plot_pairplot(df),
        "correlation":   plot_correlation_heatmap(df),
        "comparison":    plot_model_comparison(results, best_name),
        "cv":            plot_cv_comparison(results),
        "confusion":     plot_confusion_matrix(best["cm"], best_name),
        "importance":    plot_feature_importance(importances),
    }
    print(f"       {len(images)} charts rendered")

    # ── 4. Report ─────────────────────────────────────────────────────────────
    print(f"\n  [4/4] Building HTML report …")
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
    print(f"  ✅  Done in {elapsed:.1f}s")
    print(f"  📄  Report: {output_path}\n")


if __name__ == "__main__":
    main()
