"""
Activation-space similarity utilities.

Provides cosine-similarity matrices and helper functions for preparing
data that feeds into heatmap visualisations.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


def cosine_similarity_matrix(activations: np.ndarray) -> np.ndarray:
    """
    Compute an NxN cosine-similarity matrix from a set of activation vectors.

    Args:
        activations: 2-D array of shape ``(N, d)`` — one row per prompt.

    Returns:
        Symmetric NumPy array of shape ``(N, N)`` with values in ``[-1, 1]``.
        Diagonal entries are 1.0.
    """
    if activations.ndim != 2:
        raise ValueError(
            f"Expected a 2-D array of shape (N, d), got shape {activations.shape}."
        )

    # L2-normalise each row
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    normed = activations / norms  # (N, d)

    sim = normed @ normed.T  # (N, N)
    # Clip to [-1, 1] to handle floating-point rounding
    sim = np.clip(sim, -1.0, 1.0)
    return sim.astype(np.float32)


# ---------------------------------------------------------------------------
# Heatmap data preparation
# ---------------------------------------------------------------------------


def compute_feature_activation_heatmap_data(
    all_results: List[dict],
    top_n_features: int = 30,
) -> tuple[np.ndarray, List[str], List[int]]:
    """
    Prepare a (prompts x features) activation matrix for heatmap plotting.

    Only the ``top_n_features`` most frequently active features (across all
    prompts) are included.  Activation values are taken from
    ``"full_activations"`` if present, otherwise they are reconstructed from
    ``"top_k_indices"`` / ``"top_k_values"`` with zeros for non-top-k entries.

    Args:
        all_results:    Flat list of extraction result dicts.
        top_n_features: How many features to include in the heatmap (columns).

    Returns:
        ``(heatmap_matrix, prompt_labels, feature_indices)`` where:

        - ``heatmap_matrix``  is shape ``(N_prompts, top_n_features)`` float32
        - ``prompt_labels``   is a list of truncated prompt strings
        - ``feature_indices`` is the list of feature-index integers (columns)
    """
    if not all_results:
        return np.zeros((0, 0), dtype=np.float32), [], []

    # ---- determine top-n feature indices by frequency ----------------
    freq: Counter = Counter()
    for r in all_results:
        freq.update(r.get("top_k_indices", []))

    top_feature_indices: List[int] = [idx for idx, _ in freq.most_common(top_n_features)]

    if not top_feature_indices:
        return np.zeros((len(all_results), 0), dtype=np.float32), [], []

    feat_pos = {fidx: pos for pos, fidx in enumerate(top_feature_indices)}
    n_prompts = len(all_results)
    n_features = len(top_feature_indices)

    matrix = np.zeros((n_prompts, n_features), dtype=np.float32)

    for row_i, result in enumerate(all_results):
        full_acts = result.get("full_activations", [])

        if full_acts:
            full_arr = np.array(full_acts, dtype=np.float32)
            for col_j, feat_idx in enumerate(top_feature_indices):
                if feat_idx < len(full_arr):
                    matrix[row_i, col_j] = full_arr[feat_idx]
        else:
            # Fallback: use sparse top-k info
            indices = result.get("top_k_indices", [])
            values = result.get("top_k_values", [])
            for idx, val in zip(indices, values):
                if idx in feat_pos:
                    matrix[row_i, feat_pos[idx]] = float(val)

    # ---- prompt labels -----------------------------------------------
    labels = []
    for r in all_results:
        prompt = r.get("prompt", "")
        # Truncate long prompts for readability in the plot
        label = (prompt[:40] + "...") if len(prompt) > 40 else prompt
        labels.append(label)

    logger.debug(
        "Heatmap data: %d prompts x %d features", n_prompts, n_features
    )
    return matrix, labels, top_feature_indices


def compute_family_averaged_heatmap_data(
    all_results: List[dict],
    top_n_features: int = 25,
) -> tuple[np.ndarray, List[str], List[int], List[int]]:
    """
    Prepare a (families x features) activation matrix by averaging across
    all prompts in each family.

    Families are ordered by category: capitals first, then inventors, then
    world-fact families. Within each category they are sorted alphabetically
    by family_id.

    Args:
        all_results:    Flat list of extraction result dicts (from features.json).
        top_n_features: Number of most-frequent features to include as columns.

    Returns:
        ``(matrix, family_labels, feature_indices, category_boundaries)`` where:

        - ``matrix``              shape ``(N_families, top_n_features)`` float32
        - ``family_labels``       readable topic strings, one per row
        - ``feature_indices``     feature-index integers (columns)
        - ``category_boundaries`` row indices where a new category starts
          (after the first), for drawing dividing lines in the heatmap
    """
    if not all_results:
        return np.zeros((0, 0), dtype=np.float32), [], [], []

    # ---- determine top-n feature indices by frequency across ALL prompts ---
    freq: Counter = Counter()
    for r in all_results:
        freq.update(r.get("top_k_indices", []))

    top_feature_indices: List[int] = [idx for idx, _ in freq.most_common(top_n_features)]
    if not top_feature_indices:
        return np.zeros((0, 0), dtype=np.float32), [], [], []

    feat_pos = {fidx: pos for pos, fidx in enumerate(top_feature_indices)}

    # ---- group records by family_id ----------------------------------------
    from collections import defaultdict
    family_records: dict = defaultdict(list)
    family_topic: dict = {}
    for r in all_results:
        fid = r.get("family_id", "unknown")
        family_records[fid].append(r)
        if fid not in family_topic:
            family_topic[fid] = r.get("topic", fid)

    # ---- sort families by category: capitals → inventors → world facts ------
    def _category_order(fid: str) -> tuple:
        if fid.startswith("capital"):
            return (0, fid)
        if fid.startswith("inventor"):
            return (1, fid)
        return (2, fid)

    sorted_families = sorted(family_records.keys(), key=_category_order)

    # ---- build category boundary list (row index where new category begins) -
    category_boundaries: List[int] = []
    prev_cat = _category_order(sorted_families[0])[0]
    for i, fid in enumerate(sorted_families[1:], start=1):
        cat = _category_order(fid)[0]
        if cat != prev_cat:
            category_boundaries.append(i)
            prev_cat = cat

    # ---- build averaged matrix ----------------------------------------------
    n_families = len(sorted_families)
    n_features = len(top_feature_indices)
    matrix = np.zeros((n_families, n_features), dtype=np.float32)

    for row_i, fid in enumerate(sorted_families):
        records = family_records[fid]
        row_acc = np.zeros(n_features, dtype=np.float32)
        for r in records:
            indices = r.get("top_k_indices", [])
            values = r.get("top_k_values", [])
            for idx, val in zip(indices, values):
                if idx in feat_pos:
                    row_acc[feat_pos[idx]] += float(val)
        matrix[row_i] = row_acc / max(len(records), 1)

    family_labels = [family_topic[fid] for fid in sorted_families]

    logger.debug(
        "Family-averaged heatmap: %d families x %d features, boundaries at %s",
        n_families, n_features, category_boundaries,
    )
    return matrix, family_labels, top_feature_indices, category_boundaries
