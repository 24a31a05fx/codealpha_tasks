"""
visualiser.py
-------------
All matplotlib / seaborn visualisation logic.
Each function returns a base-64 PNG string — no file I/O here.
The HTML report assembles these strings.
"""

import io, base64, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import (
    FEATURES, FEATURE_DISPLAY, TARGET,
    PALETTE, COLOR_BG, COLOR_CARD, COLOR_BORDER,
    COLOR_MUTED, COLOR_BEST, COLOR_BASE, COLOR_ACCENT,
    PLOT_DPI,
)

warnings.filterwarnings("ignore")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=PLOT_DPI)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return data


def _dark_axes(ax: plt.Axes) -> None:
    """Apply consistent dark-theme styling to an axes object."""
    ax.set_facecolor(COLOR_CARD)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_BORDER)
    ax.xaxis.label.set_color(COLOR_MUTED)
    ax.yaxis.label.set_color(COLOR_MUTED)
    ax.title.set_color("white")


def _dark_fig(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(COLOR_BG)


# ── Public plot functions ─────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> str:
    """Donut chart showing per-class sample counts."""
    counts = df[TARGET].value_counts()
    labels = [s.replace("Iris-", "") for s in counts.index]

    fig, ax = plt.subplots(figsize=(5, 4))
    _dark_fig(fig); _dark_axes(ax)

    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        autopct="%1.0f%%",
        colors=PALETTE,
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=COLOR_BG, linewidth=2),
        textprops={"color": "white", "fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color(COLOR_BG)
        at.set_fontweight("bold")

    ax.set_title("Class Distribution", color="white", fontsize=12, pad=12)
    centre = plt.Circle((0, 0), 0.40, color=COLOR_CARD)
    ax.add_artist(centre)
    ax.text(0, 0, f"{len(df)}\nsamples", ha="center", va="center",
            color="white", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_feature_distributions(df: pd.DataFrame) -> str:
    """2×2 histograms — one per feature, coloured by species."""
    species_list = sorted(df[TARGET].unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    _dark_fig(fig)

    for ax, feat in zip(axes.flat, FEATURES):
        _dark_axes(ax)
        for sp, col in zip(species_list, PALETTE):
            vals = df[df[TARGET] == sp][feat]
            ax.hist(vals, alpha=0.72, color=col,
                    label=sp.replace("Iris-", ""), bins=13, edgecolor=COLOR_BG)
        ax.set_title(FEATURE_DISPLAY[feat], fontsize=10)
        ax.set_xlabel("Value (cm)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7.5, facecolor=COLOR_CARD, labelcolor="white",
                  framealpha=0.8, edgecolor=COLOR_BORDER)

    fig.suptitle("Feature Distributions by Species",
                 color="white", fontsize=14, y=1.02)
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_boxplots(df: pd.DataFrame) -> str:
    """Side-by-side box plots for each feature grouped by species."""
    plot_df = df.copy()
    plot_df[TARGET] = plot_df[TARGET].str.replace("Iris-", "")
    feat_labels = [FEATURE_DISPLAY[f] for f in FEATURES]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=False)
    _dark_fig(fig)
    species_order = ["setosa", "versicolor", "virginica"]
    pal = dict(zip(species_order, PALETTE))

    for ax, feat, label in zip(axes, FEATURES, feat_labels):
        _dark_axes(ax)
        sns.boxplot(
            data=plot_df, x=TARGET, y=feat,
            order=species_order, palette=pal,
            width=0.55, linewidth=1.2, ax=ax,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=COLOR_MUTED),
            capprops=dict(color=COLOR_MUTED),
            flierprops=dict(marker="o", markerfacecolor=COLOR_ACCENT,
                            markersize=4, alpha=0.6),
        )
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("cm", fontsize=8)
        ax.tick_params(axis="x", labelsize=8, rotation=10)

    fig.suptitle("Feature Spread by Species (Box Plots)",
                 color="white", fontsize=13, y=1.02)
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_pairplot(df: pd.DataFrame) -> str:
    """Seaborn pairplot — scatter matrix with KDE diagonal."""
    pair_df = df.copy()
    pair_df[TARGET] = pair_df[TARGET].str.replace("Iris-", "")
    pair_df.rename(columns=FEATURE_DISPLAY, inplace=True)
    species_order = ["setosa", "versicolor", "virginica"]
    pal = dict(zip(species_order, PALETTE))

    g = sns.pairplot(
        pair_df, hue=TARGET, palette=pal,
        diag_kind="kde", plot_kws={"alpha": 0.55, "s": 28},
        diag_kws={"fill": True, "alpha": 0.4},
        hue_order=species_order, height=2.1,
    )
    g.fig.patch.set_facecolor(COLOR_BG)
    for ax in g.axes.flat:
        if ax:
            _dark_axes(ax)
            ax.xaxis.label.set_fontsize(7.5)
            ax.yaxis.label.set_fontsize(7.5)

    g._legend.set_frame_on(True)
    g._legend.get_frame().set_facecolor(COLOR_CARD)
    g._legend.get_frame().set_edgecolor(COLOR_BORDER)
    for t in g._legend.get_texts():
        t.set_color("white"); t.set_fontsize(9)
    g._legend.get_title().set_color(COLOR_MUTED)

    g.fig.suptitle("Pairwise Feature Relationships",
                   color="white", fontsize=13, y=1.02)
    return _fig_to_b64(g.fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    """Pearson correlation heatmap for numeric features."""
    corr = df[FEATURES].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    _dark_fig(fig); _dark_axes(ax)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)   # upper triangle only
    sns.heatmap(
        corr, ax=ax, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, linecolor=COLOR_BG,
        annot_kws={"size": 9, "weight": "bold"},
        xticklabels=[FEATURE_DISPLAY[f] for f in FEATURES],
        yticklabels=[FEATURE_DISPLAY[f] for f in FEATURES],
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=12)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_model_comparison(results: dict, best_name: str) -> str:
    """Horizontal bar chart comparing test-set accuracy across all models."""
    names = list(results.keys())
    accs  = [results[n]["acc"] * 100 for n in names]
    colors = [COLOR_BEST if n == best_name else COLOR_BASE for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    _dark_fig(fig); _dark_axes(ax)

    bars = ax.barh(names, accs, color=colors, height=0.52,
                   edgecolor=COLOR_BG, linewidth=0.8)
    for bar, val in zip(bars, accs):
        ax.text(
            min(val + 0.3, 100.5),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left",
            color="white", fontweight="bold", fontsize=9,
        )

    ax.set_xlim(80, 102)
    ax.set_title("Model Accuracy — Test Set", fontsize=12)
    ax.set_xlabel("Accuracy (%)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.tick_params(axis="x", colors=COLOR_MUTED)
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_cv_comparison(results: dict) -> str:
    """Grouped bar chart: CV mean ± std for all models."""
    names    = list(results.keys())
    cv_means = [results[n]["cv"].mean() * 100 for n in names]
    cv_stds  = [results[n]["cv"].std()  * 100 for n in names]
    x        = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    _dark_fig(fig); _dark_axes(ax)

    bars = ax.bar(x, cv_means, color=COLOR_BASE, width=0.5,
                  yerr=cv_stds, capsize=5,
                  error_kw=dict(ecolor=COLOR_ACCENT, lw=2, capthick=2),
                  edgecolor=COLOR_BG, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.replace(" ", "\n") for n in names],
        color=COLOR_MUTED, fontsize=8.5
    )
    ax.set_ylim(84, 104)
    ax.set_title(f"5-Fold Cross-Validation Accuracy (mean ± std)", fontsize=12)
    ax.set_ylabel("CV Accuracy (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    for xi, m, s in zip(x, cv_means, cv_stds):
        ax.text(xi, m + s + 0.5, f"{m:.1f}%",
                ha="center", color="white", fontsize=8.5)
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_confusion_matrix(cm: np.ndarray, model_name: str) -> str:
    """Annotated heatmap confusion matrix."""
    labels = ["setosa", "versicolor", "virginica"]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    _dark_fig(fig); _dark_axes(ax)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.8, linecolor=COLOR_BG,
        annot_kws={"size": 14, "weight": "bold"},
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title(f"Confusion Matrix\n{model_name}", fontsize=11)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    plt.tight_layout()
    return _fig_to_b64(fig)


def plot_feature_importance(importances: dict) -> str:
    """Horizontal bar chart of Random Forest feature importances."""
    feats = list(importances.keys())[::-1]
    vals  = list(importances.values())[::-1]
    labels = [FEATURE_DISPLAY.get(f, f) for f in feats]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    _dark_fig(fig); _dark_axes(ax)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(feats))]
    bars = ax.barh(labels, vals, color=colors, height=0.48,
                   edgecolor=COLOR_BG, linewidth=0.8)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", color="white", fontsize=9)

    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_title("Feature Importance (Random Forest)", fontsize=12)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    return _fig_to_b64(fig)
