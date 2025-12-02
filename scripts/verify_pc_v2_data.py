"""Verify PC v2 data loading and splitting works correctly.

Quick sanity check before running full training.
"""

from pathlib import Path
import sys

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_pc_v2 import PCDataBuilder
from sklearn.model_selection import StratifiedKFold


def main():
    """Verify data loading."""
    print("=" * 60)
    print("PC v2 Data Verification")
    print("=" * 60)

    # Paths
    labels_path = Path("data/processed/labels_pc.jsonl")
    posts_path = Path("data/redsm5/redsm5_posts.csv")
    criteria_path = Path("data/DSM5/MDD_Criteira.json")

    # Check files exist
    print("\n1. Checking file paths...")
    for path in [labels_path, posts_path, criteria_path]:
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {path}")
        if not path.exists():
            print(f"    ERROR: File not found!")
            return False

    # Load data
    print("\n2. Loading data...")
    try:
        builder = PCDataBuilder(
            labels_path=labels_path,
            posts_path=posts_path,
            criteria_path=criteria_path,
        )
        print(f"  ✓ Loaded {len(builder.labels)} PC labels")
        print(f"  ✓ Loaded {len(builder.post_lookup)} posts")
        print(f"  ✓ Loaded {len(builder.crit_lookup)} criteria: {sorted(builder.crit_lookup.keys())}")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        return False

    # Get post IDs
    post_ids = sorted(list(set(label["post_id"] for label in builder.labels)))
    print(f"\n3. Total unique posts: {len(post_ids)}")

    # Build examples for all posts
    print("\n4. Building examples...")
    try:
        examples = builder.build_examples(post_ids)
        print(f"  ✓ Built {len(examples)} examples")

        # Check class distribution
        n_pos = sum(1 for ex in examples if ex.label == 1)
        n_neg = len(examples) - n_pos
        pos_ratio = n_pos / len(examples)
        print(f"  ✓ Class distribution: {n_pos} pos ({pos_ratio:.1%}), {n_neg} neg ({1-pos_ratio:.1%})")
    except Exception as e:
        print(f"  ✗ Error building examples: {e}")
        return False

    # Test stratified splitting
    print("\n5. Testing stratified 5-fold split...")
    try:
        # Create label patterns
        pairs_and_labels = []
        for label_rec in builder.labels:
            post_id = label_rec["post_id"]
            cid = label_rec["cid"]
            label = label_rec["label"]
            pairs_and_labels.append(((post_id, cid), label))

        pairs_and_labels.sort(key=lambda x: (x[0][0], x[0][1]))

        post_to_label_vec = {}
        for post_id in post_ids:
            post_labels = [
                label for (pid, _cid), label in pairs_and_labels if pid == post_id
            ]
            post_to_label_vec[post_id] = "".join(map(str, sorted(post_labels)))

        strat_labels = [post_to_label_vec[pid] for pid in post_ids]

        # Split
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(kfold.split(post_ids, strat_labels)):
            train_posts = [post_ids[i] for i in train_idx]
            test_posts = [post_ids[i] for i in test_idx]

            # Further split train into train + dev
            n_dev = max(1, len(train_posts) // 10)
            dev_posts = train_posts[-n_dev:]
            train_posts_final = train_posts[:-n_dev]

            print(f"  Fold {fold}: train={len(train_posts_final)}, dev={len(dev_posts)}, test={len(test_posts)}")

        print("  ✓ Stratified splitting works correctly")
    except Exception as e:
        print(f"  ✗ Error during splitting: {e}")
        return False

    # Test example building for one fold
    print("\n6. Testing example building for fold 0...")
    try:
        fold_0_train_idx, fold_0_test_idx = list(kfold.split(post_ids, strat_labels))[0]
        fold_0_train_posts = [post_ids[i] for i in fold_0_train_idx]
        fold_0_test_posts = [post_ids[i] for i in fold_0_test_idx]

        train_examples = builder.build_examples(fold_0_train_posts)
        test_examples = builder.build_examples(fold_0_test_posts)

        print(f"  ✓ Fold 0 train examples: {len(train_examples)}")
        print(f"  ✓ Fold 0 test examples: {len(test_examples)}")

        # Check one example
        if train_examples:
            ex = train_examples[0]
            print(f"\n  Sample example:")
            print(f"    post_id: {ex.post_id}")
            print(f"    cid: {ex.cid}")
            print(f"    label: {ex.label}")
            print(f"    text_len: {len(ex.text)} chars")
            print(f"    criterion_len: {len(ex.criterion)} chars")
    except Exception as e:
        print(f"  ✗ Error building fold examples: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ All verification checks passed!")
    print("=" * 60)
    print("\nReady to run: python scripts/train_pc_v2.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
