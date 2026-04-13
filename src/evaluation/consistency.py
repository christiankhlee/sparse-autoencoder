"""
Paraphrase consistency evaluation.

Measures how stable sparse feature activations are across semantically
equivalent (paraphrased) prompts within each prompt family.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.prompt_families import PromptFamily
from src.features.extract_features import FeatureExtractor
from src.features.feature_stats import compute_family_overlap_matrix

logger = logging.getLogger(__name__)


def evaluate_paraphrase_consistency(
    extractor: FeatureExtractor,
    families: List[PromptFamily],
    top_k: int = 20,
) -> pd.DataFrame:
    """
    For each prompt family compute feature-stability statistics across its
    paraphrased prompts.

    Algorithm
    ---------
    1. For every family, extract top-k features for each paraphrase.
    2. Build the NxN pairwise Jaccard-overlap matrix.
    3. Summarise: mean, std, min, max overlap (upper-triangle only).

    Args:
        extractor: Initialised :class:`~src.features.extract_features.FeatureExtractor`.
        families:  List of :class:`~src.data.prompt_families.PromptFamily` objects.
        top_k:     Number of top features per prompt to compare.

    Returns:
        :class:`pandas.DataFrame` with columns:

        - ``family_id``    – unique family identifier
        - ``topic``        – human-readable topic
        - ``n_prompts``    – number of paraphrases evaluated
        - ``mean_overlap`` – mean pairwise Jaccard similarity
        - ``std_overlap``  – standard deviation of pairwise Jaccard similarities
        - ``min_overlap``  – minimum pairwise Jaccard similarity
        - ``max_overlap``  – maximum pairwise Jaccard similarity
    """
    rows = []

    for family in tqdm(families, desc="Evaluating consistency", unit="family"):
        paraphrases = family.paraphrases
        n = len(paraphrases)

        if n < 2:
            logger.warning(
                "Family '%s' has only %d paraphrase(s); skipping consistency eval.",
                family.family_id,
                n,
            )
            continue

        logger.debug("Extracting features for family '%s' (%d prompts)", family.family_id, n)
        results = extractor.extract_for_prompts(paraphrases)

        # Build pairwise matrix
        matrix = compute_family_overlap_matrix(results)

        # Collect upper-triangle values (excludes diagonal)
        upper_vals = [
            float(matrix[i, j])
            for i in range(n)
            for j in range(i + 1, n)
        ]

        if not upper_vals:
            mean_ov = std_ov = min_ov = max_ov = 0.0
        else:
            mean_ov = float(np.mean(upper_vals))
            std_ov = float(np.std(upper_vals))
            min_ov = float(np.min(upper_vals))
            max_ov = float(np.max(upper_vals))

        rows.append(
            {
                "family_id": family.family_id,
                "topic": family.topic,
                "n_prompts": n,
                "mean_overlap": mean_ov,
                "std_overlap": std_ov,
                "min_overlap": min_ov,
                "max_overlap": max_ov,
            }
        )

        logger.info(
            "Family '%s': mean_overlap=%.3f  std=%.3f",
            family.family_id,
            mean_ov,
            std_ov,
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mean_overlap", ascending=False).reset_index(drop=True)

    logger.info("Consistency evaluation complete for %d families.", len(df))
    return df
