"""Utilities module."""

import torch


def enable_performance_optimizations():
    """Enable PyTorch performance optimizations.

    This should be called once at application startup to enable:
    - TF32 for Ampere+ GPUs (A100, RTX 3090, etc.) - 10-20% speedup
    - cuDNN auto-tuner for optimal convolution algorithms

    These optimizations provide performance improvements with minimal
    accuracy impact (TF32 provides ~10 bits of precision vs 23 for FP32).

    Returns:
        None

    Examples:
        >>> from Project.utils import enable_performance_optimizations
        >>> enable_performance_optimizations()
        >>> # Training code here...
    """
    if torch.cuda.is_available():
        # Enable TF32 for Ampere+ GPUs (compute capability >= 8.0)
        # TF32 provides 8x faster matrix multiplications with minimal accuracy loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN auto-tuner to find optimal algorithms
        # Adds overhead on first run but speeds up subsequent runs
        torch.backends.cudnn.benchmark = True
