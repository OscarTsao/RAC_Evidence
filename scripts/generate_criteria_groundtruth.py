#!/usr/bin/env python3
"""
Generate criteria matching NLI groundtruth dataset.

This script creates a dataset for Natural Language Inference (NLI) task
where each sample is a (post, DSM5_criterion) pair with a binary label
indicating whether the post matches the criterion.

Output: data/groundtruth/criteria_matching_groundtruth.csv
Total samples: 14840 (1484 posts × 10 criteria)
"""

import pandas as pd
import json
import os
from pathlib import Path


# Mapping from criterion IDs to symptom names used in annotations
CRITERION_TO_SYMPTOM = {
    'A.1': 'DEPRESSED_MOOD',
    'A.2': 'ANHEDONIA',
    'A.3': 'APPETITE_CHANGE',
    'A.4': 'SLEEP_ISSUES',
    'A.5': 'PSYCHOMOTOR',
    'A.6': 'FATIGUE',
    'A.7': 'WORTHLESSNESS',
    'A.8': 'COGNITIVE_ISSUES',
    'A.9': 'SUICIDAL_THOUGHTS',
    'A.10': 'SPECIAL_CASE'
}


def load_posts(posts_path):
    """Load all posts from redsm5_posts.csv."""
    print(f"Loading posts from {posts_path}...")
    df = pd.read_csv(posts_path)
    print(f"Loaded {len(df)} posts")
    return df


def load_criteria(criteria_path):
    """Load DSM5 criteria from MDD_Criteira.json."""
    print(f"Loading criteria from {criteria_path}...")
    with open(criteria_path, 'r', encoding='utf-8') as f:
        criteria_data = json.load(f)

    # Extract criteria list
    criteria = criteria_data.get('criteria', [])
    print(f"Loaded {len(criteria)} criteria")
    return criteria


def load_annotations(annotations_path):
    """
    Load annotations and create a set of (post_id, DSM5_symptom) pairs
    where status=1.
    """
    print(f"Loading annotations from {annotations_path}...")
    df = pd.read_csv(annotations_path)

    # Filter for status=1 and create set of positive pairs
    positive_pairs = set()
    for _, row in df[df['status'] == 1].iterrows():
        positive_pairs.add((row['post_id'], row['DSM5_symptom']))

    print(f"Found {len(positive_pairs)} positive (post_id, criterion) pairs")
    return positive_pairs


def generate_groundtruth(posts_df, criteria, positive_pairs, output_path):
    """
    Generate all combinations of posts and criteria with groundtruth labels.

    Args:
        posts_df: DataFrame with post_id and text columns
        criteria: List of criterion dictionaries with 'id' and 'text'
        positive_pairs: Set of (post_id, DSM5_symptom) tuples where status=1
        output_path: Path to save the output CSV
    """
    print("Generating groundtruth dataset...")

    # Prepare output data
    rows = []

    # Generate all combinations
    for _, post_row in posts_df.iterrows():
        post_id = post_row['post_id']
        post_text = post_row['text']

        for criterion in criteria:
            criterion_id = criterion['id']

            # Map criterion ID to symptom name for matching
            symptom_name = CRITERION_TO_SYMPTOM.get(criterion_id, criterion_id)

            # Check if this pair is in positive annotations
            is_positive = (post_id, symptom_name) in positive_pairs
            groundtruth = 1 if is_positive else 0

            rows.append({
                'post_id': post_id,
                'post': post_text,
                'DSM5_symptom': criterion_id,
                'groundtruth': groundtruth
            })

    # Create DataFrame
    result_df = pd.DataFrame(rows)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    result_df.to_csv(output_path, index=False)

    # Print statistics
    total_samples = len(result_df)
    positive_samples = (result_df['groundtruth'] == 1).sum()
    negative_samples = (result_df['groundtruth'] == 0).sum()

    print("\nDataset generated successfully!")
    print(f"Total samples: {total_samples}")
    print(f"Positive samples: {positive_samples} ({100*positive_samples/total_samples:.2f}%)")
    print(f"Negative samples: {negative_samples} ({100*negative_samples/total_samples:.2f}%)")
    print(f"Saved to: {output_path}")

    return result_df


def main():
    """Main execution function."""
    # Define paths
    base_dir = Path(__file__).parent.parent
    posts_path = base_dir / 'data' / 'redsm5' / 'redsm5_posts.csv'
    criteria_path = base_dir / 'data' / 'DSM5' / 'MDD_Criteira.json'
    annotations_path = base_dir / 'data' / 'redsm5' / 'redsm5_annotations.csv'
    output_path = base_dir / 'data' / 'groundtruth' / 'criteria_matching_groundtruth.csv'

    # Validate input files exist
    for path in [posts_path, criteria_path, annotations_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load data
    posts_df = load_posts(posts_path)
    criteria = load_criteria(criteria_path)
    positive_pairs = load_annotations(annotations_path)

    # Generate groundtruth
    generate_groundtruth(posts_df, criteria, positive_pairs, output_path)


if __name__ == '__main__':
    main()
