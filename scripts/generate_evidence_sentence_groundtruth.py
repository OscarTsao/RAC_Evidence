#!/usr/bin/env python3
"""
Generate evidence binding groundtruth for sentence-level binary classification.

This script creates a dataset where each sentence from annotated posts is
paired with each criterion, and labeled 1 if it's the evidence sentence
for that criterion, 0 otherwise.

Output: data/groundtruth/evidence_sentence_groundtruth.csv
"""

import pandas as pd
import json
import os
import re
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


def split_sentences(text):
    """
    Split text into sentences using simple punctuation-based splitting.

    Args:
        text: Input text string

    Returns:
        List of sentences
    """
    # Split by periods, question marks, and exclamation marks
    # Keep the punctuation with the sentence
    sentences = re.split(r'([.!?]+\s+)', text)

    # Recombine sentences with their punctuation
    result = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i].strip()
        if sentence:
            # Check if next element is punctuation
            if i + 1 < len(sentences) and re.match(r'^[.!?]+\s*$', sentences[i + 1]):
                sentence = sentence + sentences[i + 1].strip()
                i += 2
            else:
                i += 1
            result.append(sentence)
        else:
            i += 1

    return result


def load_posts(posts_path):
    """Load posts from redsm5_posts.csv."""
    print(f"Loading posts from {posts_path}...")
    df = pd.read_csv(posts_path)
    print(f"Loaded {len(df)} posts")
    return df


def load_criteria(criteria_path):
    """Load DSM5 criteria from MDD_Criteira.json."""
    print(f"Loading criteria from {criteria_path}...")
    with open(criteria_path, 'r', encoding='utf-8') as f:
        criteria_data = json.load(f)

    criteria = criteria_data.get('criteria', [])
    print(f"Loaded {len(criteria)} criteria")
    return criteria


def load_annotations(annotations_path):
    """
    Load annotations and create mappings for evidence sentences.

    Returns:
        annotated_posts: set of post_ids that appear in annotations
        evidence_sentences: set of (post_id, DSM5_symptom, sentence_id)
        evidence_by_post: dict[post_id] -> set of sentence_ids used in evidence
        evidence_text_lookup: dict[(post_id, DSM5_symptom, sentence_id)] -> sentence_text
        evidence_text_by_post_idx: dict[post_id] -> dict[sentence_id] -> sentence_text
    """
    print(f"Loading annotations from {annotations_path}...")
    df = pd.read_csv(annotations_path)

    evidence_sentences = set()
    annotated_posts = set()
    evidence_by_post = {}
    evidence_text_lookup = {}
    evidence_text_by_post_idx = {}

    for _, row in df.iterrows():
        post_id = row['post_id']
        annotated_posts.add(post_id)

        if row['status'] == 1:
            # Extract sentence_id from the sentence_id column (e.g., "s_1270_9_6" -> 6)
            sentence_id_str = row['sentence_id']
            parts = sentence_id_str.split('_')
            if len(parts) >= 4:
                try:
                    sentence_idx = int(parts[-1])
                    symptom = row['DSM5_symptom']
                    evidence_sentences.add((post_id, symptom, sentence_idx))

                    evidence_by_post.setdefault(post_id, set()).add(sentence_idx)
                    evidence_text_lookup[(post_id, symptom, sentence_idx)] = row['sentence_text']
                    evidence_text_by_post_idx.setdefault(post_id, {}).setdefault(
                        sentence_idx, row['sentence_text']
                    )
                except ValueError:
                    print(f"Warning: Could not parse sentence_id: {sentence_id_str}")

    print(f"Found {len(annotated_posts)} annotated posts")
    print(f"Found {len(evidence_sentences)} evidence sentences")

    return annotated_posts, evidence_sentences, evidence_by_post, evidence_text_lookup, evidence_text_by_post_idx


def generate_groundtruth(
    posts_df,
    criteria,
    annotated_posts,
    evidence_sentences,
    evidence_by_post,
    evidence_text_lookup,
    evidence_text_by_post_idx,
    output_path,
):
    """
    Generate sentence-level evidence groundtruth.

    Args:
        posts_df: DataFrame with post_id and text columns
        criteria: List of criterion dictionaries
        annotated_posts: Set of post_ids that appear in annotations
        evidence_sentences: Set of (post_id, criterion, sentence_id) tuples
        evidence_by_post: Dict mapping post_id to sentence_ids referenced by annotations
        evidence_text_lookup: Dict mapping (post_id, symptom, sentence_id) to sentence_text
        evidence_text_by_post_idx: Dict mapping post_id -> sentence_id -> sentence_text
        output_path: Path to save output CSV
    """
    print("Generating sentence-level evidence groundtruth...")

    rows = []

    # Filter to only annotated posts
    annotated_posts_df = posts_df[posts_df['post_id'].isin(annotated_posts)]
    print(f"Processing {len(annotated_posts_df)} annotated posts...")

    # Process each post
    for _, post_row in annotated_posts_df.iterrows():
        post_id = post_row['post_id']
        post_text = post_row['text']

        # Split into sentences
        sentences = split_sentences(post_text)

        # Determine how many sentences we need, padding to include any annotated indices
        max_needed_idx = max(
            len(sentences) - 1,
            max(evidence_by_post.get(post_id, [-1]), default=-1),
        )

        # For each sentence (real or padded) and criterion combination
        for sentence_id in range(max_needed_idx + 1):
            if sentence_id < len(sentences):
                sentence = sentences[sentence_id]
            else:
                # Use annotated sentence text if available; otherwise blank placeholder
                sentence = evidence_text_by_post_idx.get(post_id, {}).get(sentence_id, "")

            for criterion in criteria:
                criterion_id = criterion['id']

                # Map criterion ID to symptom name for matching
                symptom_name = CRITERION_TO_SYMPTOM.get(criterion_id, criterion_id)

                # Check if this is an evidence sentence
                is_evidence = (post_id, symptom_name, sentence_id) in evidence_sentences
                groundtruth = 1 if is_evidence else 0

                evidence_sent_text = None
                if is_evidence:
                    evidence_sent_text = evidence_text_lookup.get(
                        (post_id, symptom_name, sentence_id)
                    )
                    evidence_sent_id = sentence_id
                else:
                    evidence_sent_id = None

                rows.append({
                    'post_id': post_id,
                    'post': post_text,
                    'sentence_id': sentence_id,
                    'sentence': sentence,
                    'evidence_sentence_id': evidence_sent_id,
                    'evidence_sentence': evidence_sent_text,
                    'criterion': criterion_id,
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
    unique_posts = result_df['post_id'].nunique()

    print("\nDataset generated successfully!")
    print(f"Total samples: {total_samples}")
    print(f"Unique posts: {unique_posts}")
    print(f"Positive samples (evidence): {positive_samples} ({100*positive_samples/total_samples:.2f}%)")
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
    output_path = base_dir / 'data' / 'groundtruth' / 'evidence_sentence_groundtruth.csv'

    # Validate input files exist
    for path in [posts_path, criteria_path, annotations_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load data
    posts_df = load_posts(posts_path)
    criteria = load_criteria(criteria_path)
    (
        annotated_posts,
        evidence_sentences,
        evidence_by_post,
        evidence_text_lookup,
        evidence_text_by_post_idx,
    ) = load_annotations(annotations_path)

    # Generate groundtruth
    generate_groundtruth(
        posts_df,
        criteria,
        annotated_posts,
        evidence_sentences,
        evidence_by_post,
        evidence_text_lookup,
        evidence_text_by_post_idx,
        output_path,
    )


if __name__ == '__main__':
    main()
