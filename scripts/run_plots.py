#!/usr/bin/env python3
"""
Generate all three publication-quality figures from saved results.

Outputs are written to reports/figures/.

Usage:
    python scripts/run_plots.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.evaluation.similarity import (
    compute_family_averaged_heatmap_data,
    compute_feature_activation_heatmap_data,
)
from src.utils.io import ensure_dir, load_config, load_json
from src.utils.seed import set_seed
from src.visualization.plots import (
    plot_correct_vs_incorrect_features,
    plot_family_feature_heatmap,
    plot_feature_activation_heatmap,
    plot_overlap_by_family,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all result figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
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

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    fig_dir = Path(config["visualization"]["output_dir"])
    dpi = config["visualization"]["dpi"]
    ensure_dir(fig_dir)

    # ---- Figure 1: Overlap by family ------------------------------------
    consistency_path = Path("reports") / "consistency_results.csv"
    if consistency_path.exists():
        consistency_df = pd.read_csv(consistency_path)
        plot_overlap_by_family(
            consistency_df=consistency_df,
            output_path=str(fig_dir / "fig1_overlap_by_family.png"),
            dpi=dpi,
        )
        logger.info("Figure 1 written.")
    else:
        logger.warning(
            "consistency_results.csv not found. Run run_consistency_eval.py first."
        )

    # ---- Figure 2: Family-averaged feature activation heatmap -----------
    features_path = Path("data") / "processed" / "features.json"
    if features_path.exists():
        all_results = load_json(features_path)
        matrix, family_labels, feature_indices, category_boundaries = (
            compute_family_averaged_heatmap_data(all_results, top_n_features=25)
        )
        plot_family_feature_heatmap(
            matrix=matrix,
            family_labels=family_labels,
            feature_indices=feature_indices,
            category_boundaries=category_boundaries,
            output_path=str(fig_dir / "fig2_feature_heatmap.png"),
            dpi=dpi,
        )
        logger.info("Figure 2 written.")
    else:
        logger.warning("features.json not found. Run run_feature_extraction.py first.")

    # ---- Figure 3: Correct vs incorrect features ------------------------
    behavior_path = Path("reports") / "behavior_results.json"
    if behavior_path.exists():
        behavior_results = load_json(behavior_path)
        plot_correct_vs_incorrect_features(
            behavior_results=behavior_results,
            output_path=str(fig_dir / "fig3_correct_vs_incorrect.png"),
            top_n=20,
            dpi=dpi,
        )
        logger.info("Figure 3 written.")
    else:
        logger.warning("behavior_results.json not found. Run run_behavior_eval.py first.")

    print(f"\nAll figures saved to: {fig_dir.resolve()}\n")


if __name__ == "__main__":
    main()
