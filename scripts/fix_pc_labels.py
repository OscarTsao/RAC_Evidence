"""Fix PC labels by replacing with ground truth from criteria_matching_groundtruth.csv"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Project.utils.logging import get_logger


# Mapping from DSM5 symptoms to criterion IDs
SYMPTOM_TO_CID = {
    "A.1": "c1",  # Depressed Mood
    "A.2": "c2",  # Anhedonia
    "A.3": "c3",  # Appetite Change
    "A.4": "c4",  # Sleep Issues
    "A.5": "c5",  # Psychomotor
    "A.6": "c6",  # Fatigue
    "A.7": "c7",  # Worthlessness
    "A.8": "c8",  # Cognitive Issues
    "A.9": "c9",  # Suicidal Thoughts
    # A.10 excluded - special case not in our criteria
}


def main():
    logger = get_logger(__name__)

    # Paths
    gt_path = Path("data/groundtruth/criteria_matching_groundtruth.csv")
    old_labels_path = Path("data/processed/labels_pc.jsonl")
    new_labels_path = Path("data/processed/labels_pc_groundtruth.jsonl")
    backup_path = Path("data/processed/labels_pc_old.jsonl")

    # Load ground truth
    logger.info(f"Loading ground truth from {gt_path}")
    df = pd.read_csv(gt_path)
    logger.info(f"Loaded {len(df)} ground truth samples")

    # Convert DSM5_symptom to cid
    df["cid"] = df["DSM5_symptom"].map(SYMPTOM_TO_CID)

    # Filter out A.10 (special case)
    df = df[df["cid"].notna()].copy()
    logger.info(f"Filtered to {len(df)} samples (excluded A.10)")

    # Create labels in correct format
    labels = []
    for _, row in df.iterrows():
        labels.append({
            "post_id": row["post_id"],
            "cid": row["cid"],
            "label": int(row["groundtruth"]),
        })

    logger.info(f"Created {len(labels)} labels")

    # Check label distribution
    label_counts = pd.Series([label["label"] for label in labels]).value_counts()
    logger.info(f"Label distribution:")
    for label, count in sorted(label_counts.items()):
        logger.info(f"  {label}: {count} ({count/len(labels)*100:.2f}%)")

    # Backup old labels
    if old_labels_path.exists():
        logger.info(f"Backing up old labels to {backup_path}")
        old_labels_path.rename(backup_path)

    # Write new labels
    logger.info(f"Writing corrected labels to {new_labels_path}")
    with open(new_labels_path, "w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")

    # Also write to the original path to replace broken labels
    logger.info(f"Writing corrected labels to {old_labels_path}")
    with open(old_labels_path, "w") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")

    logger.info("Done!")
    logger.info(f"\nSummary:")
    logger.info(f"  Ground truth samples: {len(df)}")
    logger.info(f"  Labels created: {len(labels)}")
    logger.info(f"  Old labels backed up: {backup_path}")
    logger.info(f"  New labels written: {old_labels_path}")


if __name__ == "__main__":
    main()
