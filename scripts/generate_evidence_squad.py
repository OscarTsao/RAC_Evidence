#!/usr/bin/env python3
"""
Generate evidence binding groundtruth in SQuAD format.

This script creates a dataset for span extraction where each sample
contains a post, criterion, and the character-level start/end positions
of the evidence sentence within the post.

Output: data/groundtruth/evidence_squad_groundtruth.csv
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


def split_sentences_with_positions(text):
    """
    Split text into sentences and return their character positions.

    Args:
        text: Input text string

    Returns:
        List of tuples (sentence_text, start_idx, end_idx)
    """
    # Split by periods, question marks, and exclamation marks
    pattern = r'([.!?]+\s+)'
    parts = re.split(pattern, text)

    sentences = []
    current_pos = 0
    i = 0

    while i < len(parts):
        part = parts[i]

        if part.strip():
            sentence_text = part
            start_idx = current_pos

            # Check if next element is punctuation separator
            if i + 1 < len(parts) and re.match(r'^[.!?]+\s*$', parts[i + 1]):
                sentence_text = part + parts[i + 1]
                i += 2
            else:
                i += 1

            end_idx = start_idx + len(sentence_text)

            # Store trimmed sentence with original position
            sentences.append((sentence_text.strip(), start_idx, end_idx))
            current_pos = end_idx
        else:
            current_pos += len(part)
            i += 1

    return sentences


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
    Load annotations and create mapping of (post_id, criterion) to all evidence sentences
    for status=1 cases.
    """
    print(f"Loading annotations from {annotations_path}...")
    df = pd.read_csv(annotations_path)

    # Create mapping: (post_id, DSM5_symptom) -> set of (sentence_id, sentence_text) for status=1
    evidence_mapping = {}
    positive_posts = set()

    for _, row in df[df['status'] == 1].iterrows():
        post_id = row['post_id']
        criterion = row['DSM5_symptom']

        # Extract sentence_id from the sentence_id column (e.g., "s_1270_9_6" -> 6)
        sentence_id_str = row['sentence_id']
        parts = sentence_id_str.split('_')
        if len(parts) >= 4:
            try:
                sentence_idx = int(parts[-1])
                key = (post_id, criterion)
                evidence_mapping.setdefault(key, set()).add((sentence_idx, row['sentence_text']))
                positive_posts.add(post_id)
            except ValueError:
                print(f"Warning: Could not parse sentence_id: {sentence_id_str}")

    print(f"Found {len(positive_posts)} posts with positive annotations")
    print(f"Found {sum(len(v) for v in evidence_mapping.values())} evidence sentences across "
          f"{len(evidence_mapping)} (post, criterion) pairs")

    return positive_posts, evidence_mapping


def generate_groundtruth(posts_df, criteria, positive_posts, evidence_mapping, output_path):
    """
    Generate SQuAD-style evidence groundtruth.

    Args:
        posts_df: DataFrame with post_id and text columns
        criteria: List of criterion dictionaries
        positive_posts: Set of post_ids with status=1 annotations
        evidence_mapping: Dict mapping (post_id, criterion) to sentence_id
        output_path: Path to save output CSV
    """
    print("Generating SQuAD-style evidence groundtruth...")

    rows = []
    skipped_count = 0

    # Filter to only posts with positive annotations
    positive_posts_df = posts_df[posts_df['post_id'].isin(positive_posts)]
    print(f"Processing {len(positive_posts_df)} posts with positive annotations...")

    # Process each post
    for _, post_row in positive_posts_df.iterrows():
        post_id = post_row['post_id']
        post_text = post_row['text']

        # Split into sentences with positions
        sentences_with_pos = split_sentences_with_positions(post_text)

        # For each criterion
        for criterion in criteria:
            criterion_id = criterion['id']

            # Map criterion ID to symptom name for matching
            symptom_name = CRITERION_TO_SYMPTOM.get(criterion_id, criterion_id)

            # Check if this (post, criterion) pair has evidence
            key = (post_id, symptom_name)
            evidence_entries = evidence_mapping.get(key, [])
            for sentence_id, evidence_sentence_text in evidence_entries:
                # Validate sentence_id is within range
                if sentence_id < len(sentences_with_pos):
                    sentence_text, start_idx, end_idx = sentences_with_pos[sentence_id]

                    # Verify the span is correct by extracting from original text
                    extracted = post_text[start_idx:end_idx].strip()

                    # Account for whitespace differences
                    if extracted != sentence_text:
                        # Try to find the sentence in the text
                        sentence_clean = sentence_text.strip()
                        idx = post_text.find(sentence_clean)
                        if idx != -1:
                            start_idx = idx
                            end_idx = idx + len(sentence_clean)
                else:
                    # For out-of-range indices, try to locate the annotated text within the post
                    sentence_clean = evidence_sentence_text.strip()
                    start_idx = post_text.find(sentence_clean)
                    if start_idx != -1:
                        end_idx = start_idx + len(sentence_clean)
                    else:
                        # Secondary attempt: strip trailing punctuation
                        trimmed = sentence_clean.rstrip(".!?")
                        start_idx = post_text.find(trimmed) if trimmed else -1
                        if start_idx != -1:
                            end_idx = start_idx + len(trimmed)
                        else:
                            print(
                                f"Warning: sentence_id {sentence_id} not found in post {post_id} "
                                f"(len={len(sentences_with_pos)} sentences)"
                            )
                            skipped_count += 1
                            continue

                rows.append({
                    'post_id': post_id,
                    'post': post_text,
                    'evidence_sentence_id': sentence_id,
                    'evidence_sentence': evidence_sentence_text,
                    'criterion': criterion_id,
                    'start_idx': start_idx,
                    'end_idx': end_idx
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
    unique_posts = result_df['post_id'].nunique()
    unique_criteria = result_df['criterion'].nunique()

    print("\nDataset generated successfully!")
    print(f"Total evidence spans: {total_samples}")
    print(f"Unique posts: {unique_posts}")
    print(f"Unique criteria: {unique_criteria}")
    if skipped_count > 0:
        print(f"Skipped samples (invalid sentence_id): {skipped_count}")
    print(f"Saved to: {output_path}")

    # Validation: verify some spans
    if len(result_df) > 0:
        print("\nValidating random sample of spans:")
        sample_rows = result_df.sample(min(5, len(result_df)))
        for idx, row in sample_rows.iterrows():
            extracted = row['post'][row['start_idx']:row['end_idx']]
            print(f"  Post {row['post_id']}, Criterion {row['criterion']}:")
            print(f"    Extracted: {extracted[:100]}...")

    return result_df


def main():
    """Main execution function."""
    # Define paths
    base_dir = Path(__file__).parent.parent
    posts_path = base_dir / 'data' / 'redsm5' / 'redsm5_posts.csv'
    criteria_path = base_dir / 'data' / 'DSM5' / 'MDD_Criteira.json'
    annotations_path = base_dir / 'data' / 'redsm5' / 'redsm5_annotations.csv'
    output_path = base_dir / 'data' / 'groundtruth' / 'evidence_squad_groundtruth.csv'

    # Validate input files exist
    for path in [posts_path, criteria_path, annotations_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # Load data
    posts_df = load_posts(posts_path)
    criteria = load_criteria(criteria_path)
    positive_posts, evidence_mapping = load_annotations(annotations_path)

    # Generate groundtruth
    generate_groundtruth(posts_df, criteria, positive_posts, evidence_mapping, output_path)


if __name__ == '__main__':
    main()
