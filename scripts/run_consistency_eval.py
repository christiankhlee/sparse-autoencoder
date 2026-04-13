#!/usr/bin/env python3
"""
Evaluate paraphrase consistency from saved features.json.

Loads pre-computed top-k feature indices and computes pairwise Jaccard
overlap across paraphrases within each prompt family. No model or SAE
inference is needed — everything comes from the saved features file.

Usage:
    python scripts/run_consistency_eval.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.features.feature_stats import compute_jaccard_overlap
from src.utils.io import ensure_dir, load_config, load_json
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate paraphrase feature consistency from saved features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
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

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    # ---- Load saved features --------------------------------------------
    features_path = Path(config["data"]["processed_dir"]) / "features.json"
    if not features_path.exists():
        logger.error(
            "features.json not found: %s\nRun run_feature_extraction.py first.",
            features_path,
        )
        sys.exit(1)

    all_results = load_json(features_path)
    logger.info("Loaded %d feature records from %s", len(all_results), features_path)

    # ---- Group paraphrase records by family -----------------------------
    family_paraphrases: dict[str, list[dict]] = defaultdict(list)
    family_meta: dict[str, str] = {}

    for r in all_results:
        if r.get("is_paraphrase", False):
            fid = r["family_id"]
            family_paraphrases[fid].append(r)
            family_meta[fid] = r.get("topic", fid)

    # ---- Compute pairwise Jaccard per family ----------------------------
    rows = []
    for fid, records in sorted(family_paraphrases.items()):
        n = len(records)
        if n < 2:
            logger.warning("Family '%s' has only %d paraphrase(s); skipping.", fid, n)
            continue

        upper_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                idx_i = np.array(records[i].get("top_k_indices", []))
                idx_j = np.array(records[j].get("top_k_indices", []))
                upper_vals.append(compute_jaccard_overlap(idx_i, idx_j))

        rows.append({
            "family_id": fid,
            "topic": family_meta[fid],
            "n_prompts": n,
            "mean_overlap": float(np.mean(upper_vals)),
            "std_overlap": float(np.std(upper_vals)),
            "min_overlap": float(np.min(upper_vals)),
            "max_overlap": float(np.max(upper_vals)),
        })
        logger.info(
            "Family '%s': mean_overlap=%.3f  std=%.3f",
            fid, rows[-1]["mean_overlap"], rows[-1]["std_overlap"],
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("mean_overlap", ascending=False)
        .reset_index(drop=True)
    )

    # ---- Save -----------------------------------------------------------
    out_dir = Path("reports")
    ensure_dir(out_dir)
    out_path = out_dir / "consistency_results.csv"
    df.to_csv(out_path, index=False)
    logger.info("Consistency results saved to %s", out_path)

    print("\n" + "=" * 60)
    print("  Consistency Evaluation Summary")
    print("=" * 60)
    print(df.to_string(index=False, float_format="{:.3f}".format))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
