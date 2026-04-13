#!/usr/bin/env python3
"""
Load GPT-2 and a pretrained SAE, then extract sparse features for every
prompt in the dataset.  Results are saved to data/processed/features.json.

Usage:
    python scripts/run_feature_extraction.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
torch.set_num_threads(1)  # prevent macOS semaphore leaks from PyTorch thread pool

from tqdm import tqdm

from src.data.prompt_families import PromptFamily
from src.features.extract_features import FeatureExtractor
from src.features.load_sae import load_pretrained_sae
from src.models.load_model import load_model_and_tokenizer
from src.utils.io import ensure_dir, load_config, load_json, save_json
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract sparse features for all prompts in the dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # ---- Config ----------------------------------------------------------
    logger.info("Loading config from: %s", args.config)
    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    # ---- Load prompt families -------------------------------------------
    families_path = Path(config["data"]["prompt_families_file"])
    if not families_path.exists():
        logger.error(
            "Prompt families file not found: %s\n"
            "Run scripts/run_dataset.py first.",
            families_path,
        )
        sys.exit(1)

    raw_families = load_json(families_path)
    families = [PromptFamily.from_dict(d) for d in raw_families]
    logger.info("Loaded %d prompt families.", len(families))

    # ---- Load model & SAE -----------------------------------------------
    device = config["model"]["device"]
    model, tokenizer = load_model_and_tokenizer(config["model"]["name"], device)

    sae, cfg_dict = load_pretrained_sae(
        release=config["sae"]["release"],
        sae_id=config["sae"]["id"],
        device=device,
    )

    top_k = config["features"]["top_k"]
    layer_idx = config["sae"]["layer"]

    extractor = FeatureExtractor(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        layer_idx=layer_idx,
        device=device,
        top_k=top_k,
    )

    # ---- Extract features (save after each family so crashes don't lose work) ---
    out_path = Path(config["data"]["processed_dir"]) / "features.json"
    ensure_dir(out_path.parent)

    # Resume from existing results if present
    all_results: list[dict] = []
    done_families: set[str] = set()
    if out_path.exists():
        all_results = load_json(out_path)
        done_families = {r["family_id"] for r in all_results}
        logger.info("Resuming: %d families already done: %s", len(done_families), sorted(done_families))

    for family in tqdm(families, desc="Families", unit="family"):
        if family.family_id in done_families:
            logger.info("Skipping already-done family: %s", family.family_id)
            continue

        all_prompts = family.paraphrases + family.incorrect_prompts
        logger.info(
            "Processing family '%s' (%d prompts) ...",
            family.family_id, len(all_prompts),
        )

        prompt_results = extractor.extract_for_prompts(all_prompts)

        # Tag each result with metadata
        for i, res in enumerate(prompt_results):
            res["family_id"] = family.family_id
            res["topic"] = family.topic
            res["correct_answer"] = family.correct_answer
            res["is_paraphrase"] = i < len(family.paraphrases)

        all_results.extend(prompt_results)

        # Save after every family so a crash doesn't lose previous work
        save_json(all_results, out_path)
        logger.info("Saved %d records so far -> %s", len(all_results), out_path)

    n_para = sum(1 for r in all_results if r.get("is_paraphrase"))
    n_dist = len(all_results) - n_para
    logger.info(
        "Done. %d total records (%d paraphrases, %d distractors) in %s",
        len(all_results), n_para, n_dist, out_path,
    )
    print(f"\nFeature extraction complete. Results: {out_path}\n")


if __name__ == "__main__":
    main()
