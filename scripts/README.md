# Groundtruth Generation Scripts

This directory contains three Python scripts for generating groundtruth datasets from the ReDSM5 annotations.

## Overview

All scripts map between DSM-5 criterion IDs (A.1-A.10) and symptom names used in annotations:

- **A.1** → DEPRESSED_MOOD
- **A.2** → ANHEDONIA  
- **A.3** → APPETITE_CHANGE
- **A.4** → SLEEP_ISSUES
- **A.5** → PSYCHOMOTOR
- **A.6** → FATIGUE
- **A.7** → WORTHLESSNESS
- **A.8** → COGNITIVE_ISSUES
- **A.9** → SUICIDAL_THOUGHTS
- **A.10** → SPECIAL_CASE

## Scripts

### 1. generate_criteria_groundtruth.py

**Purpose**: Generate criteria matching NLI (Natural Language Inference) groundtruth

**Output**: `data/groundtruth/criteria_matching_groundtruth.csv`

**Schema**: `post_id, post, DSM5_symptom, groundtruth`

**Description**: Creates all combinations of posts × criteria (1,484 posts × 10 criteria = 14,840 samples). Labels are 1 when the (post_id, DSM5_symptom) pair exists in annotations with status=1, otherwise 0.

**Statistics**:
- Total samples: 14,840
- Positive: 1,379 (9.29%)
- Negative: 13,461 (90.71%)

**Usage**:
```bash
python3 scripts/generate_criteria_groundtruth.py
```

---

### 2. generate_evidence_sentence_groundtruth.py

**Purpose**: Generate sentence-level evidence binding groundtruth for binary classification

**Output**: `data/groundtruth/evidence_sentence_groundtruth.csv`

**Schema**: `post_id, post, sentence_id, sentence, evidence_sentence_id, evidence_sentence, criterion, groundtruth`

**Description**: Only processes posts appearing in annotations. Splits each post into sentences and creates all (sentence, criterion) pairs. Labels are 1 when the sentence_id matches evidence in annotations with status=1, otherwise 0. For each (post, criterion) pair, includes the evidence sentence ID and text from annotations.

**Statistics**:
- Total samples: 298,490
- Unique posts: 1,477
- Positive: 1,507 (0.50%)
- Negative: 296,983 (99.50%)

**Usage**:
```bash
python3 scripts/generate_evidence_sentence_groundtruth.py
```

---

### 3. generate_evidence_squad.py

**Purpose**: Generate SQuAD-style evidence span extraction groundtruth

**Output**: `data/groundtruth/evidence_squad_groundtruth.csv`

**Schema**: `post_id, post, evidence_sentence_id, evidence_sentence, criterion, start_idx, end_idx`

**Description**: Only processes posts with status=1 annotations. For each (post, criterion) pair with evidence, includes the evidence sentence ID and text from annotations, along with character-level start and end positions of the evidence sentence within the post.

**Statistics**:
- Total evidence spans: 1,339
- Unique posts: 1,096
- Skipped (sentence_id out of range): 40

**Usage**:
```bash
python3 scripts/generate_evidence_squad.py
```

---

## Running All Scripts

To regenerate all groundtruth files:

```bash
cd /home/oscartsao/Developer/Template_ReDSM5

python3 scripts/generate_criteria_groundtruth.py
python3 scripts/generate_evidence_sentence_groundtruth.py
python3 scripts/generate_evidence_squad.py
```

All output files will be created in the `data/groundtruth/` directory.

## Notes

- All scripts use consistent sentence splitting based on punctuation (. ! ?)
- Sentence IDs are 0-indexed
- Character positions in SQuAD format are 0-indexed
- Scripts are idempotent and can be run multiple times
- The evidence_squad script may skip some samples where sentence splitting differs from the original annotations
