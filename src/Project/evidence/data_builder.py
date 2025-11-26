"""Evidence data builder with hybrid negative sampling.

Builds training data for sentence-criterion (SC) cross-encoder with:
- Positives from labels_sc
- Hard negatives from retrieval results (same-post Top-K)
- Random negatives from same-post sentences
- Cross-post SAFE negatives (from labels_pc == 0)
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from Project.dataio.schemas import Criterion, LabelPC, LabelSC, RetrievalSC, Sentence
from Project.utils.io import load_models
from Project.utils.logging import get_logger


@dataclass
class SCExample:
    """Sentence-criterion training example."""

    post_id: str
    sent_id: str
    cid: str
    text: str
    criterion: str
    label: int
    source: str  # 'positive', 'hard_neg', 'random_neg', 'cross_post_neg'
    retrieval_score: float | None = None


class EvidenceDataBuilder:
    """Build evidence training data with hybrid negative sampling."""

    def __init__(
        self,
        sentences: List[Sentence],
        criteria: List[Criterion],
        labels_sc: List[LabelSC],
        labels_pc: List[LabelPC],
        retrieval_sc: List[RetrievalSC] | None = None,
        k_train: int = 100,
        neg_per_pos: int = 6,
        hard_neg_ratio: float = 0.8,
        random_neg_ratio: float = 0.1,
        cross_post_ratio: float = 0.1,
        xpost_max_frac: float = 0.2,
        seed: int = 42,
        epoch_refresh: bool = False,
    ):
        """Initialize data builder.

        Args:
            sentences: All sentences
            criteria: All criteria
            labels_sc: Sentence-criterion labels
            labels_pc: Post-criterion labels
            retrieval_sc: Optional retrieval results for hard negatives
            k_train: Top-K from retrieval to use
            neg_per_pos: Number of negatives per positive (default 6 for 1:4 pos:neg ratio)
            hard_neg_ratio: Ratio of hard negatives from retrieval (default 0.8 = 80%)
            random_neg_ratio: Ratio of random same-post negatives (default 0.1 = 10%)
            cross_post_ratio: Ratio of cross-post SAFE negatives (default 0.1 = 10%)
            xpost_max_frac: Max fraction of cross-post negatives allowed (default 0.2 = 20%)
            seed: Random seed
            epoch_refresh: Whether to refresh hard negatives each epoch
        """
        self.sentences = sentences
        self.criteria = criteria
        self.labels_sc = labels_sc
        self.labels_pc = labels_pc
        self.retrieval_sc = retrieval_sc
        self.k_train = k_train
        self.neg_per_pos = neg_per_pos
        self.hard_neg_ratio = hard_neg_ratio
        self.random_neg_ratio = random_neg_ratio
        self.cross_post_ratio = cross_post_ratio
        self.xpost_max_frac = xpost_max_frac
        self.epoch_refresh = epoch_refresh
        self.seed = seed
        self.rng = random.Random(seed)
        self.logger = get_logger(__name__)
        self.retrieval_lookup: Dict[Tuple[str, str], List[Tuple[str, float, Dict]]] = {}
        if retrieval_sc:
            for record in retrieval_sc:
                ordered = sorted(
                    [
                        (str(sent_id), float(score), meta or {})
                        for sent_id, score, meta in record.candidates
                    ],
                    key=lambda item: (-item[1], item[0]),
                )
                self.retrieval_lookup[(record.post_id, record.cid)] = ordered

        # Validate ratios
        total_ratio = hard_neg_ratio + random_neg_ratio + cross_post_ratio
        if abs(total_ratio - 1.0) > 0.01:
            self.logger.warning(
                f"Negative ratios sum to {total_ratio:.2f}, not 1.0. "
                f"They will be normalized."
            )

        # Build lookups
        self.sent_lookup = {(s.post_id, s.sent_id): s for s in sentences}
        self.crit_lookup = {c.cid: c for c in criteria}
        self.sents_by_post: Dict[str, List[Sentence]] = defaultdict(list)
        for s in sentences:
            self.sents_by_post[s.post_id].append(s)

        # Build positive set
        self.positives: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for lab in labels_sc:
            if lab.label == 1:
                self.positives[(lab.post_id, lab.cid)].add(lab.sent_id)

        # Build cross-post SAFE pool (posts where criterion is negative)
        self.cross_post_safe: Dict[str, Set[str]] = defaultdict(set)
        for lab in labels_pc:
            if lab.label == 0:
                self.cross_post_safe[lab.cid].add(lab.post_id)

    def refresh_seed(self, epoch: int) -> None:
        """Refresh random seed for epoch-wise hard negative sampling.

        Args:
            epoch: Current epoch number
        """
        if self.epoch_refresh:
            new_seed = self.seed + epoch
            self.rng = random.Random(new_seed)
            self.logger.info(f"Refreshed random seed to {new_seed} for epoch {epoch}")

    def build_examples(
        self, post_ids: List[str] | None = None, epoch: int = 0
    ) -> List[SCExample]:
        """Build training examples with hybrid negatives.

        Args:
            post_ids: Optional list of post IDs to include (for fold splits)
            epoch: Current epoch (used for epoch-wise refresh if enabled)

        Returns:
            List of SC examples
        """
        # Refresh seed if epoch-wise refresh is enabled
        self.refresh_seed(epoch)
        if post_ids is None:
            post_ids = list(set(lab.post_id for lab in self.labels_sc))

        post_ids_set = set(post_ids)
        examples = []

        for post_id in post_ids:
            post_sents = self.sents_by_post.get(post_id, [])
            if not post_sents:
                continue

            # Group positives by criterion
            for cid in self.crit_lookup.keys():
                pos_sents = self.positives.get((post_id, cid), set())
                # In training, we skip if no positives for this (post, crit)
                # This might be a design choice: do we train on purely negative examples?
                # The original code skipped:
                if not pos_sents:
                    continue  # Skip if no positives for this (post, crit)

                crit_desc = self.crit_lookup[cid].desc

                # Add positives
                for sent_id in pos_sents:
                    sent = self.sent_lookup.get((post_id, sent_id))
                    if sent:
                        examples.append(
                            SCExample(
                                post_id=post_id,
                                sent_id=sent_id,
                                cid=cid,
                                text=sent.text,
                                criterion=crit_desc,
                                label=1,
                                source="positive",
                            )
                        )

                # Sample negatives
                n_pos = len(pos_sents)
                n_neg = self.neg_per_pos * n_pos

                # Split negatives by type
                n_hard = int(n_neg * self.hard_neg_ratio)
                n_random = int(n_neg * self.random_neg_ratio)
                n_cross_desired = int(n_neg * self.cross_post_ratio)

                # Enforce xpost_max_frac cap
                n_cross_max = int(n_neg * self.xpost_max_frac)
                n_cross = min(n_cross_desired, n_cross_max)

                # Adjust random to fill any gap if cross-post was capped
                if n_cross < n_cross_desired:
                    shortage = n_cross_desired - n_cross
                    n_random += shortage

                negatives = []

                # 1. Hard negatives from retrieval (Top-K same-post)
                if self.retrieval_sc and n_hard > 0:
                    hard_negs = self._sample_hard_negatives(
                        post_id, cid, pos_sents, n_hard
                    )
                    negatives.extend(hard_negs)

                # 2. Random negatives from same-post
                if n_random > 0:
                    random_negs = self._sample_random_negatives(
                        post_id, cid, pos_sents, n_random, post_sents
                    )
                    negatives.extend(random_negs)

                # 3. Cross-post SAFE negatives
                if n_cross > 0:
                    cross_negs = self._sample_cross_post_negatives(
                        post_id, cid, n_cross, allowed_posts=post_ids_set
                    )
                    negatives.extend(cross_negs)

                # Add negative examples
                for sent, source in negatives:
                    examples.append(
                        SCExample(
                            post_id=post_id,
                            sent_id=sent.sent_id,
                            cid=cid,
                            text=sent.text,
                            criterion=crit_desc,
                            label=0,
                            source=source,
                        )
                    )

        # Log statistics
        n_pos = sum(1 for e in examples if e.label == 1)
        n_neg = sum(1 for e in examples if e.label == 0)
        n_hard = sum(1 for e in examples if e.source == "hard_neg")
        n_random = sum(1 for e in examples if e.source == "random_neg")
        n_cross = sum(1 for e in examples if e.source == "cross_post_neg")

        self.logger.info(
            f"Built {len(examples)} examples: {n_pos} pos, {n_neg} neg "
            f"(hard={n_hard}, random={n_random}, cross-post={n_cross})"
        )
        if n_neg > 0:
            self.logger.info(
                f"Negative distribution: "
                f"hard={n_hard / n_neg:.1%}, "
                f"random={n_random / n_neg:.1%}, "
                f"cross-post={n_cross / n_neg:.1%}"
            )

        return examples

    def build_inference_examples(
        self,
        post_ids: List[str],
        k_infer: int = 20,
        criteria_ids: List[str] | None = None,
    ) -> List[SCExample]:
        """Build inference candidates from same-post retrieval results.

        Args:
            post_ids: Posts to include (typically dev/test fold)
            k_infer: Top-K per (post, cid) to keep
            criteria_ids: Optional list of criterion IDs to restrict to

        Returns:
            List of SC examples covering retrieval candidates
        """
        if not self.retrieval_lookup:
            raise ValueError(
                "Retrieval results are required to build inference examples. "
                "Provide retrieval_sc when constructing EvidenceDataBuilder."
            )

        criteria_order = criteria_ids or list(self.crit_lookup.keys())
        examples: List[SCExample] = []

        for post_id in post_ids:
            for cid in criteria_order:
                retrieval_key = (post_id, cid)
                candidates = self.retrieval_lookup.get(retrieval_key, [])
                if not candidates:
                    continue

                crit_desc = self.crit_lookup[cid].desc
                positive_sents = self.positives.get(retrieval_key, set())

                for sent_id, score, _meta in candidates[:k_infer]:
                    sent = self.sent_lookup.get((post_id, sent_id))
                    if not sent:
                        self.logger.debug(
                            "Skipping missing sentence (post=%s, sent=%s, cid=%s)",
                            post_id,
                            sent_id,
                            cid,
                        )
                        continue

                    examples.append(
                        SCExample(
                            post_id=post_id,
                            sent_id=sent.sent_id,
                            cid=cid,
                            text=sent.text,
                            criterion=crit_desc,
                            label=1 if sent_id in positive_sents else 0,
                            source="retrieval",
                            retrieval_score=score,
                        )
                    )

        self.logger.info(
            "Prepared %d inference candidates across %d posts (k=%d)",
            len(examples),
            len(post_ids),
            k_infer,
        )
        return examples

    def _sample_hard_negatives(
        self, post_id: str, cid: str, pos_sents: Set[str], n: int
    ) -> List[Tuple[Sentence, str]]:
        """Sample hard negatives from retrieval Top-K."""
        candidates = []

        if self.retrieval_sc:
            # Find retrieval for this (post, cid)
            retrieval = next(
                (r for r in self.retrieval_sc if r.post_id == post_id and r.cid == cid),
                None,
            )

            if retrieval:
                # Get Top-K candidates excluding positives
                for cand in retrieval.candidates[: self.k_train]:
                    sent_id = cand[0]
                    if sent_id not in pos_sents:
                        sent = self.sent_lookup.get((post_id, sent_id))
                        if sent:
                            candidates.append(sent)

        # Sample n candidates
        self.rng.shuffle(candidates)
        return [(s, "hard_neg") for s in candidates[:n]]

    def _sample_random_negatives(
        self,
        post_id: str,
        cid: str,
        pos_sents: Set[str],
        n: int,
        post_sents: List[Sentence],
    ) -> List[Tuple[Sentence, str]]:
        """Sample random negatives from same-post sentences."""
        candidates = [s for s in post_sents if s.sent_id not in pos_sents]
        self.rng.shuffle(candidates)
        return [(s, "random_neg") for s in candidates[:n]]

    def _sample_cross_post_negatives(
        self, post_id: str, cid: str, n: int, allowed_posts: Set[str]
    ) -> List[Tuple[Sentence, str]]:
        """Sample cross-post SAFE negatives (from posts where criterion is negative)."""
        candidates = []

        # Get posts where this criterion is negative
        safe_posts = self.cross_post_safe.get(cid, set())
        safe_posts = [p for p in safe_posts if p != post_id and p in allowed_posts]

        # Sample sentences from safe posts
        for safe_post in safe_posts:
            post_sents = self.sents_by_post.get(safe_post, [])
            candidates.extend(post_sents)

        # Sample n candidates
        self.rng.shuffle(candidates)
        return [(s, "cross_post_neg") for s in candidates[:n]]


def load_evidence_data(
    raw_dir: Path,
    interim_dir: Path,
    k_train: int = 100,
    neg_per_pos: int = 6,
    hard_neg_ratio: float = 0.8,
    random_neg_ratio: float = 0.1,
    cross_post_ratio: float = 0.1,
    xpost_max_frac: float = 0.2,
    seed: int = 42,
    epoch_refresh: bool = False,
) -> EvidenceDataBuilder:
    """Load data and create evidence data builder.

    Args:
        raw_dir: Directory with raw data (sentences, criteria, labels)
        interim_dir: Directory with retrieval results
        k_train: Top-K from retrieval to use
        neg_per_pos: Number of negatives per positive (default 6)
        hard_neg_ratio: Ratio of hard negatives (default 0.8 = 80%)
        random_neg_ratio: Ratio of random negatives (default 0.1 = 10%)
        cross_post_ratio: Ratio of cross-post SAFE negatives (default 0.1 = 10%)
        xpost_max_frac: Max fraction of cross-post negatives (default 0.2 = 20%)
        seed: Random seed
        epoch_refresh: Whether to refresh hard negatives each epoch

    Returns:
        EvidenceDataBuilder instance
    """
    from Project.utils.data import load_raw_dataset

    logger = get_logger(__name__)

    # Load raw data
    data = load_raw_dataset(raw_dir)

    # Load retrieval results
    retrieval_path = interim_dir / "retrieval_sc.jsonl"
    retrieval_sc = None
    if retrieval_path.exists():
        retrieval_sc = load_models(retrieval_path, RetrievalSC)
        logger.info(f"Loaded {len(retrieval_sc)} retrieval results from {retrieval_path}")
    else:
        logger.warning(f"No retrieval results found at {retrieval_path}")

    # Create data builder
    builder = EvidenceDataBuilder(
        sentences=data["sentences"],
        criteria=data["criteria"],
        labels_sc=data["labels_sc"],
        labels_pc=data["labels_pc"],
        retrieval_sc=retrieval_sc,
        k_train=k_train,
        neg_per_pos=neg_per_pos,
        hard_neg_ratio=hard_neg_ratio,
        random_neg_ratio=random_neg_ratio,
        cross_post_ratio=cross_post_ratio,
        xpost_max_frac=xpost_max_frac,
        seed=seed,
        epoch_refresh=epoch_refresh,
    )

    logger.info(
        f"Created EvidenceDataBuilder with k_train={k_train}, "
        f"neg_per_pos={neg_per_pos}, ratios=({hard_neg_ratio:.0%}/"
        f"{random_neg_ratio:.0%}/{cross_post_ratio:.0%}), "
        f"xpost_max={xpost_max_frac:.0%} (strict guard)"
    )

    return builder
