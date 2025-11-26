"""Generate post-level 5-fold splits for Evidence pipeline.

Creates stratified post-level splits ensuring:
- No leakage: A post never appears in its own training/calibration fold
- Deterministic: Same seed produces same splits
- Balanced: Attempts to balance criterion distribution across folds
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from Project.dataio.schemas import LabelSC
from Project.utils.io import load_models
from Project.utils.logging import get_logger


def create_post_level_5fold(
    labels_sc: List[LabelSC],
    n_folds: int = 5,
    seed: int = 42,
) -> Dict:
    """Create post-level 5-fold splits.

    Args:
        labels_sc: Sentence-criterion labels
        n_folds: Number of folds (default 5)
        seed: Random seed for reproducibility

    Returns:
        Dict with fold information
    """
    logger = get_logger(__name__)
    np.random.seed(seed)

    # Get all unique posts
    all_posts = sorted(set(lab.post_id for lab in labels_sc))
    n_posts = len(all_posts)

    logger.info(f"Creating {n_folds}-fold splits for {n_posts} posts")

    # Count criteria per post for stratification
    post_criteria_counts: Dict[str, int] = defaultdict(int)
    for lab in labels_sc:
        if lab.label == 1:  # Only count positives
            post_criteria_counts[lab.post_id] += 1

    # Sort posts by criteria count (for stratification)
    posts_sorted = sorted(all_posts, key=lambda p: post_criteria_counts.get(p, 0))

    # Simple round-robin assignment for stratification
    fold_assignments = {}
    for i, post_id in enumerate(posts_sorted):
        fold_assignments[post_id] = i % n_folds

    # Create fold splits
    folds = []
    for fold_idx in range(n_folds):
        train_posts = [p for p, f in fold_assignments.items() if f != fold_idx]
        test_posts = [p for p, f in fold_assignments.items() if f == fold_idx]

        fold_info = {
            "fold": fold_idx,
            "train_post_ids": sorted(train_posts),
            "test_post_ids": sorted(test_posts),
            "n_train": len(train_posts),
            "n_test": len(test_posts),
        }

        folds.append(fold_info)

        # Log statistics
        train_positives = sum(
            1
            for lab in labels_sc
            if lab.post_id in train_posts and lab.label == 1
        )
        test_positives = sum(
            1 for lab in labels_sc if lab.post_id in test_posts and lab.label == 1
        )

        logger.info(
            f"Fold {fold_idx}: train={len(train_posts)} posts ({train_positives} pos), "
            f"test={len(test_posts)} posts ({test_positives} pos)"
        )

    # Leakage check
    for fold in folds:
        train_set = set(fold["train_post_ids"])
        test_set = set(fold["test_post_ids"])
        overlap = train_set & test_set

        if overlap:
            logger.error(f"Fold {fold['fold']}: Leakage detected! {len(overlap)} posts overlap")
            raise ValueError(f"Data leakage in fold {fold['fold']}")

    logger.info(f"✓ No leakage detected across {n_folds} folds")

    return {
        "n_folds": n_folds,
        "n_posts_total": n_posts,
        "seed": seed,
        "folds": folds,
    }


def main():
    """CLI entry point for split generation."""
    parser = argparse.ArgumentParser(
        description="Generate post-level 5-fold splits for Evidence pipeline"
    )
    parser.add_argument(
        "--labels_sc",
        required=True,
        help="Path to labels_sc.jsonl",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds (default 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for splits JSON",
    )

    args = parser.parse_args()

    logger = get_logger(__name__)

    # Load labels
    logger.info(f"Loading labels from {args.labels_sc}")
    labels_sc = load_models(Path(args.labels_sc), LabelSC)
    logger.info(f"Loaded {len(labels_sc)} labels")

    # Create splits
    splits = create_post_level_5fold(
        labels_sc=labels_sc,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    # Save to file
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(f"Saved splits to {output_path}")
    logger.info(f"✓ Created {splits['n_folds']} folds for {splits['n_posts_total']} posts")


if __name__ == "__main__":
    main()
