"""
Statistical utilities for sparse feature analysis.

Functions here operate on the output of :class:`~src.features.extract_features.FeatureExtractor`
(lists of dicts with ``top_k_indices`` / ``top_k_values`` / ``full_activations``).
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pair-wise overlap metrics
# ---------------------------------------------------------------------------


def compute_jaccard_overlap(set_a: np.ndarray, set_b: np.ndarray) -> float:
    """
    Jaccard similarity between two arrays of feature indices.

    .. math::
        J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}

    Args:
        set_a: 1-D array of integer feature indices.
        set_b: 1-D array of integer feature indices.

    Returns:
        Jaccard similarity in ``[0, 1]``.  Returns ``0.0`` if both sets
        are empty.
    """
    a = set(set_a.tolist() if isinstance(set_a, np.ndarray) else set_a)
    b = set(set_b.tolist() if isinstance(set_b, np.ndarray) else set_b)

    union = a | b
    if not union:
        return 0.0

    intersection = a & b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Family-level overlap
# ---------------------------------------------------------------------------


def compute_family_overlap_matrix(family_results: List[dict]) -> np.ndarray:
    """
    Build an NxN pairwise Jaccard-overlap matrix for a single prompt family.

    Args:
        family_results: List of extraction dicts (output of
            ``FeatureExtractor.extract_for_prompts``).  Each dict must
            contain a ``"top_k_indices"`` key.

    Returns:
        Symmetric NumPy array of shape ``(N, N)`` with diagonal entries 1.0.
    """
    n = len(family_results)
    matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            elif j > i:
                idx_i = np.array(family_results[i]["top_k_indices"])
                idx_j = np.array(family_results[j]["top_k_indices"])
                score = compute_jaccard_overlap(idx_i, idx_j)
                matrix[i, j] = score
                matrix[j, i] = score  # symmetric

    return matrix


def compute_mean_family_stability(family_results: List[dict]) -> float:
    """
    Mean pairwise Jaccard overlap — a scalar "stability score" for a family.

    Off-diagonal entries only (we exclude self-similarity = 1.0).

    Args:
        family_results: As in :func:`compute_family_overlap_matrix`.

    Returns:
        Mean Jaccard overlap over all non-identical prompt pairs.
        Returns ``0.0`` for a singleton family.
    """
    n = len(family_results)
    if n < 2:
        return 0.0

    matrix = compute_family_overlap_matrix(family_results)

    # Collect upper-triangle (excluding diagonal)
    upper = [matrix[i, j] for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(upper)) if upper else 0.0


# ---------------------------------------------------------------------------
# Global feature frequency ranking
# ---------------------------------------------------------------------------


def rank_features_by_frequency(all_results: List[dict]) -> pd.DataFrame:
    """
    Rank features by how often they appear in the top-k set across all prompts.

    Args:
        all_results: Flat list of extraction dicts (from any number of
            prompt families / families combined).

    Returns:
        :class:`pandas.DataFrame` with columns:

        - ``feature_idx``  – integer feature index
        - ``frequency``    – number of prompts where this feature was in top-k
        - ``frac``         – fraction of prompts (``frequency / n_prompts``)
        - ``mean_value``   – mean activation value when active
        - ``rank``         – 1-based rank (most frequent = 1)
    """
    counter: Counter = Counter()
    value_accum: dict[int, list[float]] = {}

    for result in all_results:
        indices = result.get("top_k_indices", [])
        values = result.get("top_k_values", [])
        for idx, val in zip(indices, values):
            counter[idx] += 1
            value_accum.setdefault(idx, []).append(float(val))

    n_prompts = len(all_results)

    rows = []
    for feat_idx, freq in counter.most_common():
        rows.append(
            {
                "feature_idx": feat_idx,
                "frequency": freq,
                "frac": freq / max(n_prompts, 1),
                "mean_value": float(np.mean(value_accum[feat_idx])),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("frequency", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df
