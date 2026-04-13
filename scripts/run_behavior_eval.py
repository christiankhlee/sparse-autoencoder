#!/usr/bin/env python3
"""
Evaluate the association between sparse features and model behavior.

Loads pre-computed top-k feature indices from features.json and runs only
next-token prediction (model only, no SAE) to determine correct vs incorrect
outputs. The SAE is NOT reloaded here.

Results are saved to reports/behavior_results.json.

Usage:
    python scripts/run_behavior_eval.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
torch.set_num_threads(1)

from tqdm import tqdm

from src.models.load_model import load_model_and_tokenizer
from src.utils.io import ensure_dir, load_config, load_json, save_json
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate behavior–feature associations from saved features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


@torch.no_grad()
def predict_next_token(model, tokenizer, prompt: str, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model(**inputs)
    token_id = outputs.logits[0, -1, :].argmax(dim=-1).item()
    return tokenizer.decode(token_id)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    # ---- Load saved features --------------------------------------------
    features_path = Path(config["data"]["processed_dir"]) / "features.json"
    if not features_path.exists():
        logger.error("features.json not found. Run run_feature_extraction.py first.")
        sys.exit(1)

    all_results = load_json(features_path)
    # Index by prompt text for fast lookup
    features_by_prompt: dict[str, dict] = {r["prompt"]: r for r in all_results}
    logger.info("Loaded %d feature records.", len(all_results))

    # ---- Load model only (no SAE needed for predictions) ----------------
    device = config["model"]["device"]
    model, tokenizer = load_model_and_tokenizer(config["model"]["name"], device)

    # ---- Load prompt families for correct_answer and prompt lists -------
    families_path = Path(config["data"]["prompt_families_file"])
    if not families_path.exists():
        logger.error("prompt_families.json not found. Run run_dataset.py first.")
        sys.exit(1)

    from src.data.prompt_families import PromptFamily
    families = [PromptFamily.from_dict(d) for d in load_json(families_path)]
    logger.info("Loaded %d families.", len(families))

    # ---- Run predictions and collect feature associations ---------------
    correct_counter: Counter = Counter()
    incorrect_counter: Counter = Counter()
    per_prompt_results = []

    for family in tqdm(families, desc="Behavior analysis", unit="family"):
        all_prompts = [
            (p, True) for p in family.paraphrases
        ] + [
            (p, False) for p in family.incorrect_prompts
        ]

        for prompt, is_paraphrase in all_prompts:
            predicted = predict_next_token(model, tokenizer, prompt, device)
            is_correct = predicted.strip().lower() == family.correct_answer.strip().lower()

            # Get pre-computed top-k indices
            saved = features_by_prompt.get(prompt, {})
            top_k_indices = saved.get("top_k_indices", [])
            top_k_values = saved.get("top_k_values", [])

            if is_paraphrase:
                if is_correct:
                    correct_counter.update(top_k_indices)
                else:
                    incorrect_counter.update(top_k_indices)
            else:
                # Distractors always count as incorrect context
                incorrect_counter.update(top_k_indices)

            per_prompt_results.append({
                "prompt": prompt,
                "predicted": predicted,
                "correct_answer": family.correct_answer,
                "is_correct": is_correct,
                "is_paraphrase": is_paraphrase,
                "top_k_indices": top_k_indices,
                "top_k_values": top_k_values,
                "family_id": family.family_id,
            })

    # ---- Differentiating features ---------------------------------------
    all_features = set(correct_counter.keys()) | set(incorrect_counter.keys())
    diff_rows = sorted(
        [
            {
                "feature_idx": f,
                "correct_count": correct_counter[f],
                "incorrect_count": incorrect_counter[f],
                "difference": correct_counter[f] - incorrect_counter[f],
            }
            for f in all_features
        ],
        key=lambda r: abs(r["difference"]),
        reverse=True,
    )

    n_para = sum(1 for r in per_prompt_results if r["is_paraphrase"])
    n_correct = sum(1 for r in per_prompt_results if r["is_paraphrase"] and r["is_correct"])

    behavior_results = {
        "correct_features": dict(correct_counter),
        "incorrect_features": dict(incorrect_counter),
        "differentiating_features": diff_rows,
        "per_prompt_results": per_prompt_results,
        "summary": {
            "n_families": len(families),
            "n_paraphrase_prompts": n_para,
            "n_correct_predictions": n_correct,
            "accuracy_on_paraphrases": n_correct / max(n_para, 1),
            "n_distractor_prompts": len(per_prompt_results) - n_para,
        },
    }

    logger.info(
        "Paraphrase accuracy: %d/%d (%.1f%%)",
        n_correct, n_para, 100 * behavior_results["summary"]["accuracy_on_paraphrases"],
    )

    # ---- Save -----------------------------------------------------------
    ensure_dir(Path("reports"))
    out_path = Path("reports") / "behavior_results.json"
    save_json(behavior_results, out_path)
    logger.info("Behavior results saved to %s", out_path)

    summary = behavior_results["summary"]
    print("\n" + "=" * 60)
    print("  Behavior Evaluation Summary")
    print("=" * 60)
    print(f"  Families evaluated        : {summary['n_families']}")
    print(f"  Paraphrase prompts        : {summary['n_paraphrase_prompts']}")
    print(f"  Correct predictions       : {summary['n_correct_predictions']}")
    print(f"  Accuracy on paraphrases   : {100 * summary['accuracy_on_paraphrases']:.1f}%")
    print(f"  Distractor prompts        : {summary['n_distractor_prompts']}")

    print("\n  Top 5 differentiating features:")
    for row in diff_rows[:5]:
        direction = "CORRECT" if row["difference"] > 0 else "INCORRECT"
        print(
            f"    Feature {row['feature_idx']:>6}  "
            f"correct={row['correct_count']:>3}  "
            f"incorrect={row['incorrect_count']:>3}  "
            f"diff={row['difference']:>+4}  [{direction}]"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
