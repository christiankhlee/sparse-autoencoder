"""
Behavior-feature association analysis.

Studies whether sparse feature activations differ systematically between
prompts where the model produces the correct answer and prompts where it
does not.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import List

from tqdm import tqdm

from src.data.prompt_families import PromptFamily
from src.features.extract_features import FeatureExtractor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def check_correct_prediction(prediction: str, correct_answer: str) -> bool:
    """
    Check whether the model's top-1 token matches the correct answer.

    Comparison is case-insensitive and strips leading/trailing whitespace
    from both strings.

    Args:
        prediction:     Decoded next-token string from the model.
        correct_answer: Ground-truth answer string.

    Returns:
        ``True`` if the strings match after normalisation.
    """
    return prediction.strip().lower() == correct_answer.strip().lower()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate_behavior_feature_association(
    extractor: FeatureExtractor,
    families: List[PromptFamily],
    top_k: int = 20,
) -> dict:
    """
    Compare feature patterns when the model predicts correctly vs incorrectly.

    For **paraphrase** prompts the "expected" next token is
    ``family.correct_answer``; for **incorrect** prompts the model is not
    expected to produce the correct answer (they serve as negative controls).

    Args:
        extractor: Initialised :class:`~src.features.extract_features.FeatureExtractor`.
        families:  List of :class:`~src.data.prompt_families.PromptFamily`.
        top_k:     Number of top features per prompt.

    Returns:
        Dictionary with keys:

        - ``"correct_features"``          – :class:`collections.Counter` mapping
          feature index -> number of correct-prediction prompts it appeared in.
        - ``"incorrect_features"``        – same for incorrect predictions.
        - ``"differentiating_features"``  – list of dicts
          ``{feature_idx, correct_count, incorrect_count, difference}``
          sorted by ``|difference|`` descending.
        - ``"per_prompt_results"``        – list of per-prompt dicts with keys
          ``prompt``, ``predicted``, ``correct_answer``, ``is_correct``,
          ``top_k_indices``, ``family_id``.
        - ``"summary"``                   – high-level accuracy counts.
    """
    correct_counter: Counter = Counter()
    incorrect_counter: Counter = Counter()
    per_prompt_results = []

    all_families_iter = tqdm(families, desc="Behavior analysis", unit="family")

    for family in all_families_iter:
        # ---- paraphrase prompts (expect correct answer) ----------------
        for prompt in family.paraphrases:
            result = _process_single_prompt(
                extractor,
                prompt=prompt,
                correct_answer=family.correct_answer,
                family_id=family.family_id,
                is_paraphrase=True,
            )
            per_prompt_results.append(result)

            if result["is_correct"]:
                correct_counter.update(result["top_k_indices"])
            else:
                incorrect_counter.update(result["top_k_indices"])

        # ---- incorrect / distractor prompts ----------------------------
        for prompt in family.incorrect_prompts:
            result = _process_single_prompt(
                extractor,
                prompt=prompt,
                correct_answer=family.correct_answer,
                family_id=family.family_id,
                is_paraphrase=False,
            )
            per_prompt_results.append(result)
            # These prompts are expected to produce a different token; we
            # always count them as "incorrect" for the feature comparison.
            incorrect_counter.update(result["top_k_indices"])

    # ---- differentiating features ------------------------------------
    all_features = set(correct_counter.keys()) | set(incorrect_counter.keys())
    diff_rows = []
    for feat_idx in all_features:
        c_count = correct_counter[feat_idx]
        i_count = incorrect_counter[feat_idx]
        diff_rows.append(
            {
                "feature_idx": feat_idx,
                "correct_count": c_count,
                "incorrect_count": i_count,
                "difference": c_count - i_count,
            }
        )

    diff_rows.sort(key=lambda r: abs(r["difference"]), reverse=True)

    # ---- summary -------------------------------------------------------
    n_correct = sum(1 for r in per_prompt_results if r.get("is_paraphrase") and r["is_correct"])
    n_para = sum(1 for r in per_prompt_results if r.get("is_paraphrase"))

    summary = {
        "n_families": len(families),
        "n_paraphrase_prompts": n_para,
        "n_correct_predictions": n_correct,
        "accuracy_on_paraphrases": n_correct / max(n_para, 1),
        "n_distractor_prompts": len(per_prompt_results) - n_para,
    }

    logger.info(
        "Behavior analysis complete. Paraphrase accuracy: %d/%d (%.1f%%)",
        n_correct,
        n_para,
        100 * summary["accuracy_on_paraphrases"],
    )

    return {
        "correct_features": dict(correct_counter),
        "incorrect_features": dict(incorrect_counter),
        "differentiating_features": diff_rows,
        "per_prompt_results": per_prompt_results,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _process_single_prompt(
    extractor: FeatureExtractor,
    prompt: str,
    correct_answer: str,
    family_id: str,
    is_paraphrase: bool,
) -> dict:
    """Run extraction + prediction for one prompt and return a result dict."""
    try:
        last_act = extractor.get_last_token_activation(prompt)
        top_indices, top_values = extractor.get_top_k_features(last_act)
        predicted = extractor.get_next_token_prediction(prompt)
        is_correct = check_correct_prediction(predicted, correct_answer)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error processing prompt %r: %s", prompt, exc)
        top_indices = []
        top_values = []
        predicted = ""
        is_correct = False

    return {
        "prompt": prompt,
        "predicted": predicted,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "is_paraphrase": is_paraphrase,
        "top_k_indices": list(top_indices) if hasattr(top_indices, "__iter__") else [],
        "top_k_values": list(top_values) if hasattr(top_values, "__iter__") else [],
        "family_id": family_id,
    }
