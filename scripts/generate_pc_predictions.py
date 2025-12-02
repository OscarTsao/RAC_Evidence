"""Generate PC predictions using ensemble of 5 fold models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from Project.dataio.schemas import Criterion, Post
from Project.utils.data import load_raw_dataset
from Project.utils.hydra_utils import load_config
from Project.utils.logging import get_logger


def load_model_and_tokenizer(model_path: Path):
    """Load a trained model and tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def predict_batch(model, tokenizer, texts: List[str], max_length: int = 512):
    """Run inference on a batch of texts."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)  # Shape: (batch_size,)
        probs = torch.sigmoid(logits)

    return logits.cpu().numpy(), probs.cpu().numpy()


def main():
    logger = get_logger(__name__)

    # Load config
    cfg = load_config("configs/ce_pc_real.yaml")
    exp = cfg.get("exp", "real_dev")

    # Paths
    raw_dir = Path(cfg.data.raw_dir)
    output_dir = Path(f"outputs/runs/{exp}/pc")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {raw_dir}")
    data = load_raw_dataset(raw_dir)
    posts: List[Post] = data["posts"]
    criteria: List[Criterion] = data["criteria"]

    logger.info(f"Loaded {len(posts)} posts and {len(criteria)} criteria")

    # Load all 5 fold models
    fold_models = []
    for fold_idx in range(5):
        fold_dir = output_dir / f"fold{fold_idx}"
        if not fold_dir.exists():
            logger.error(f"Fold {fold_idx} model not found at {fold_dir}")
            raise FileNotFoundError(f"Missing fold {fold_idx} model")

        logger.info(f"Loading fold {fold_idx} model from {fold_dir}")
        model, tokenizer = load_model_and_tokenizer(fold_dir)
        fold_models.append((model, tokenizer))

    logger.info(f"Loaded {len(fold_models)} fold models")

    # Generate predictions
    predictions = []

    for post in tqdm(posts, desc="Generating PC predictions"):
        for criterion in criteria:
            # Create input text (post + criterion)
            text = f"{post.text} [SEP] {criterion.desc}"

            # Get predictions from all 5 models
            all_logits = []
            all_probs = []

            for model, tokenizer in fold_models:
                logits, probs = predict_batch(model, tokenizer, [text])
                all_logits.append(logits[0])
                all_probs.append(probs[0])

            # Ensemble: average probabilities
            avg_prob = float(sum(all_probs) / len(all_probs))
            avg_logit = float(sum(all_logits) / len(all_logits))

            predictions.append({
                "post_id": post.post_id,
                "cid": criterion.cid,
                "logit": avg_logit,
                "prob": avg_prob,
                "ensemble_method": "average",
                "n_models": len(fold_models),
            })

    # Save predictions
    output_path = output_dir / "oof_predictions.jsonl"
    logger.info(f"Saving {len(predictions)} predictions to {output_path}")

    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    logger.info("Done!")
    logger.info(f"Generated predictions for {len(posts)} posts × {len(criteria)} criteria = {len(predictions)} total")


if __name__ == "__main__":
    main()
