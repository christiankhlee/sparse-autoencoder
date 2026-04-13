"""
Build the prompt-families dataset and persist it to disk.

Usage (called from scripts/run_dataset.py, not directly):
    from src.data.build_dataset import build_and_save_dataset
    build_and_save_dataset(config)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from src.data.prompt_families import PromptFamily, build_factual_recall_families
from src.utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)


def build_and_save_dataset(config: dict) -> List[PromptFamily]:
    """
    Build factual-recall prompt families, save to JSON, and print a summary.

    Args:
        config: Parsed configuration dictionary (from ``load_config``).

    Returns:
        The list of :class:`~src.data.prompt_families.PromptFamily` objects
        that were built and saved.
    """
    output_path = Path(config["data"]["prompt_families_file"])
    ensure_dir(output_path.parent)

    logger.info("Building factual-recall prompt families ...")
    families = build_factual_recall_families()

    # Serialise to plain dicts so they are JSON-friendly.
    serialisable = [f.to_dict() for f in families]
    save_json(serialisable, output_path)
    logger.info("Saved %d families to %s", len(families), output_path)

    # ---- summary stats -----------------------------------------------
    total_paraphrases = sum(len(f.paraphrases) for f in families)
    total_incorrect = sum(len(f.incorrect_prompts) for f in families)
    total_prompts = total_paraphrases + total_incorrect

    topics = sorted({f.topic for f in families})

    print("\n" + "=" * 60)
    print("  Dataset Summary")
    print("=" * 60)
    print(f"  Families          : {len(families)}")
    print(f"  Total paraphrases : {total_paraphrases}")
    print(f"  Incorrect prompts : {total_incorrect}")
    print(f"  Total prompts     : {total_prompts}")
    print(f"  Output file       : {output_path}")
    print("-" * 60)
    print("  Families per category:")

    categories = {
        "Capital cities": [f for f in families if f.family_id.startswith("capital_")],
        "Inventors":      [f for f in families if f.family_id.startswith("inventor_")],
        "World facts":    [
            f for f in families
            if not f.family_id.startswith("capital_")
            and not f.family_id.startswith("inventor_")
        ],
    }
    for cat, fams in categories.items():
        print(f"    {cat:<20}: {len(fams)}")

    print("-" * 60)
    print("  Family IDs:")
    for f in families:
        print(
            f"    {f.family_id:<30}  answer={f.correct_answer!r:<12}"
            f"  paraphrases={len(f.paraphrases)}"
        )
    print("=" * 60 + "\n")

    return families
