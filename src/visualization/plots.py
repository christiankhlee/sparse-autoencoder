"""
Publication-quality visualisations for sparse feature analysis.

All functions:
- Accept pre-computed data structures (no model calls inside)
- Save to PNG at the specified DPI
- Use a clean, print-friendly aesthetic
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # non-interactive backend; safe for scripts and notebooks

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

_STYLE = "seaborn-v0_8-whitegrid"
_PALETTE = "viridis"
_DPI = 150

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _ensure_parent(path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Plot 1 – Paraphrase consistency bar chart
# ---------------------------------------------------------------------------


def plot_overlap_by_family(
    consistency_df: pd.DataFrame,
    output_path: str,
    dpi: int = _DPI,
) -> None:
    """
    Bar chart of mean Jaccard overlap per prompt family, sorted descending,
    with ± std error bars.

    Args:
        consistency_df: DataFrame produced by
            :func:`~src.evaluation.consistency.evaluate_paraphrase_consistency`.
            Must contain columns ``family_id``, ``mean_overlap``, ``std_overlap``.
        output_path:    Where to write the PNG file.
        dpi:            Output resolution.
    """
    if consistency_df.empty:
        logger.warning("plot_overlap_by_family: empty DataFrame, skipping.")
        return

    df = consistency_df.sort_values("mean_overlap", ascending=False).copy()

    # Shorten labels for display
    df["label"] = df["family_id"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(df))
    colors = sns.color_palette("Blues_d", len(df))[::-1]

    bars = ax.bar(
        x,
        df["mean_overlap"],
        yerr=df["std_overlap"],
        color=colors,
        capsize=4,
        edgecolor="white",
        linewidth=0.6,
        error_kw={"elinewidth": 1.2, "ecolor": "#555555"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Mean Jaccard Overlap", fontsize=11)
    ax.set_xlabel("Prompt Family", fontsize=11)
    ax.set_title(
        "Feature Stability Across Paraphrased Prompts\n(mean ± std pairwise Jaccard overlap)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylim(0, min(1.05, df["mean_overlap"].max() + df["std_overlap"].max() + 0.15))
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Annotate each bar with its mean value
    for bar, val in zip(bars, df["mean_overlap"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )

    fig.tight_layout()
    out = _ensure_parent(output_path)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved overlap bar chart to %s", out)


# ---------------------------------------------------------------------------
# Plot 2 – Feature activation heatmap
# ---------------------------------------------------------------------------


def plot_feature_activation_heatmap(
    heatmap_matrix: np.ndarray,
    prompt_labels: List[str],
    feature_indices: List[int],
    output_path: str,
    dpi: int = _DPI,
) -> None:
    """
    Heatmap of activation strength: rows = prompts, columns = features.

    Args:
        heatmap_matrix:  2-D array of shape ``(N_prompts, N_features)``.
        prompt_labels:   Row labels (truncated prompt strings).
        feature_indices: Column labels (feature integer indices).
        output_path:     Where to write the PNG file.
        dpi:             Output resolution.
    """
    if heatmap_matrix.size == 0:
        logger.warning("plot_feature_activation_heatmap: empty matrix, skipping.")
        return

    n_prompts, n_features = heatmap_matrix.shape

    # Dynamic figure height based on number of prompts
    fig_height = max(5, n_prompts * 0.35 + 2)
    fig_width = max(10, n_features * 0.5 + 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    col_labels = [f"F{idx}" for idx in feature_indices]

    sns.heatmap(
        heatmap_matrix,
        ax=ax,
        xticklabels=col_labels,
        yticklabels=prompt_labels,
        cmap="YlOrRd",
        linewidths=0.3,
        linecolor="#dddddd",
        cbar_kws={"label": "Feature Activation", "shrink": 0.7},
        vmin=0,
    )

    ax.set_title(
        "Sparse Feature Activations Across Prompts\n"
        f"(top {n_features} most frequent features)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Feature Index", fontsize=11)
    ax.set_ylabel("Prompt", fontsize=11)
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    out = _ensure_parent(output_path)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature heatmap to %s", out)


# ---------------------------------------------------------------------------
# Plot 2b – Family-averaged feature activation heatmap
# ---------------------------------------------------------------------------


def plot_family_feature_heatmap(
    matrix: "np.ndarray",
    family_labels: List[str],
    feature_indices: List[int],
    category_boundaries: List[int],
    output_path: str,
    dpi: int = _DPI,
) -> None:
    """
    Heatmap of average activation strength: rows = prompt families,
    columns = top features.  Horizontal lines separate the three categories
    (capitals, inventors, world facts).

    Args:
        matrix:               2-D float32 array ``(N_families, N_features)``.
        family_labels:        Row labels — human-readable topic strings.
        feature_indices:      Column labels — feature integer indices.
        category_boundaries:  Row indices where a new category begins.
        output_path:          Where to write the PNG file.
        dpi:                  Output resolution.
    """
    if matrix.size == 0:
        logger.warning("plot_family_feature_heatmap: empty matrix, skipping.")
        return

    n_families, n_features = matrix.shape
    col_labels = [f"F{idx}" for idx in feature_indices]

    # Compact figure: 16 rows reads easily at ~0.45 in/row
    fig_height = max(5, n_families * 0.45 + 2.5)
    fig_width = max(10, n_features * 0.42 + 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=col_labels,
        yticklabels=family_labels,
        cmap="YlOrRd",
        linewidths=0.25,
        linecolor="#e8e8e8",
        cbar_kws={"label": "Mean Activation", "shrink": 0.65},
        vmin=0,
    )

    # Draw bold horizontal lines between categories
    for boundary in category_boundaries:
        ax.axhline(boundary, color="#333333", linewidth=1.8, zorder=5)

    ax.set_title(
        "Sparse Feature Activations by Prompt Family\n"
        f"(mean over prompts, top {n_features} features by frequency)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("SAE Feature Index", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=9, rotation=0)

    fig.tight_layout()
    out = _ensure_parent(output_path)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved family feature heatmap to %s", out)


# ---------------------------------------------------------------------------
# Plot 3 – Correct vs incorrect feature association
# ---------------------------------------------------------------------------


def plot_correct_vs_incorrect_features(
    behavior_results: dict,
    output_path: str,
    top_n: int = 20,
    dpi: int = _DPI,
) -> None:
    """
    Horizontal grouped bar chart showing which features are most associated
    with correct vs incorrect model outputs.

    Args:
        behavior_results: Dict produced by
            :func:`~src.evaluation.behavior_analysis.evaluate_behavior_feature_association`.
        output_path:       Where to write the PNG file.
        top_n:             How many differentiating features to display.
        dpi:               Output resolution.
    """
    diff_rows = behavior_results.get("differentiating_features", [])
    if not diff_rows:
        logger.warning("plot_correct_vs_incorrect_features: no data, skipping.")
        return

    # Take top-n by absolute difference
    rows = sorted(diff_rows, key=lambda r: abs(r["difference"]), reverse=True)[:top_n]

    labels = [f"F{r['feature_idx']}" for r in rows]
    correct_counts = [r["correct_count"] for r in rows]
    incorrect_counts = [r["incorrect_count"] for r in rows]

    y = np.arange(len(rows))
    bar_height = 0.38

    fig_height = max(5, len(rows) * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    bars_c = ax.barh(
        y + bar_height / 2,
        correct_counts,
        height=bar_height,
        color="#2ecc71",
        label="Correct prediction",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_i = ax.barh(
        y - bar_height / 2,
        incorrect_counts,
        height=bar_height,
        color="#e74c3c",
        label="Incorrect prediction",
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of Prompts", fontsize=11)
    ax.set_title(
        f"Top {len(rows)} Features by Correct vs Incorrect Prediction Association",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.xaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    fig.tight_layout()
    out = _ensure_parent(output_path)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correct/incorrect feature chart to %s", out)
