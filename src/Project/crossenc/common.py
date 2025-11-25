"""Shared utilities for the lightweight cross-encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import torch
from torch import amp, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from Project.dataio.loaders import PairDataset, PairExample, collate_pairwise_ce, resolve_tokenizer
from Project.utils.seed import set_seed


class CrossEncoder(nn.Module):
    """Minimal transformer-like encoder using embeddings + mean pooling."""

    def __init__(self, vocab_size: int = 60000, hidden_size: int = 128) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        mask = attention_mask.unsqueeze(-1)
        x = x * mask
        summed = x.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom
        hidden = torch.relu(self.encoder(pooled))
        return self.classifier(hidden).squeeze(-1)


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    epochs: int = 1
    batch_size: int = 8
    max_length: int = 256
    seed: int = 42
    use_amp: bool = False  # Enable mixed precision training
    amp_dtype: str = "bf16"  # bf16 | fp16 | fp32


def train_model(
    examples: Sequence[PairExample],
    cfg: TrainingConfig,
    tokenizer: Optional[object] = None,
) -> CrossEncoder:
    """Train a tiny cross-encoder on provided examples with optional mixed precision."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossEncoder().to(device)
    tk = resolve_tokenizer(tokenizer)
    ds = PairDataset(examples)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_pairwise_ce(b, tokenizer=tk, max_length=cfg.max_length),
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    bce = nn.BCEWithLogitsLoss()

    amp_dtype = cfg.amp_dtype.lower()
    use_amp = cfg.use_amp and device.type == "cuda" and amp_dtype in ("bf16", "fp16")
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    scaler = amp.GradScaler("cuda", enabled=use_amp and amp_dtype == "fp16")

    model.train()
    for _ in range(cfg.epochs):
        for batch in loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            opt.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Mixed precision forward pass
            with amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype if use_amp else None):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = bce(logits, batch["labels"])

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

    return model


def predict(
    model: CrossEncoder,
    examples: Iterable[PairExample],
    tokenizer: Optional[object],
    max_length: int,
    batch_size: int = 16,
) -> List[float]:
    model.eval()
    tk = resolve_tokenizer(tokenizer)
    ds = PairDataset(list(examples))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_pairwise_ce(b, tokenizer=tk, max_length=max_length),
    )
    preds: List[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds.extend(logits.tolist())
    return preds
