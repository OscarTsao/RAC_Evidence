"""Common cross-encoder components for inference."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoder(nn.Module):
    """Simple cross-encoder wrapper for sequence classification.

    This is a lightweight wrapper around AutoModelForSequenceClassification
    for binary classification tasks (sentence-criterion and post-criterion matching).
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 1,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def forward(self, input_ids, attention_mask):
        """Forward pass returning logits."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)


def predict(
    model: CrossEncoder,
    examples: List,
    tokenizer: Optional[AutoTokenizer] = None,
    max_length: int = 512,
    batch_size: int = 32,
) -> List[float]:
    """Run inference on examples and return logits.

    Args:
        model: CrossEncoder model instance
        examples: List of PairExample objects with .text and .criterion attributes
        tokenizer: Optional tokenizer (will create from model_name if None)
        max_length: Maximum sequence length
        batch_size: Batch size for inference

    Returns:
        List of logit scores (float)
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.model_name, use_fast=True)

    model.eval()
    all_logits = []

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]

            # Pair text with criterion
            texts = [f"{ex.text} [SEP] {ex.criterion}" for ex in batch]

            # Tokenize
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Move to device
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            # Forward pass
            logits = model(input_ids, attention_mask)
            all_logits.extend(logits.cpu().tolist())

    return all_logits
