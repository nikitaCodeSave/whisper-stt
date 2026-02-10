"""GPU memory utilities for monitoring and cleanup."""

from __future__ import annotations

import gc
import logging
import os

import torch

logger = logging.getLogger(__name__)


def log_gpu_memory(label: str) -> None:
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        logger.info("[GPU] %s: CUDA not available", label)
        return
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
    logger.info(
        "[GPU] %s: allocated=%.0fMB, reserved=%.0fMB, peak=%.0fMB",
        label, allocated, reserved, max_allocated,
    )


def cleanup_gpu_memory(label: str) -> None:
    """Run gc.collect() + empty CUDA cache, then log memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_gpu_memory(label)


def configure_cuda_allocator() -> None:
    """Set PYTORCH_CUDA_ALLOC_CONF for better memory management."""
    desired = {"max_split_size_mb": "128", "expandable_segments": "True"}
    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    existing_pairs: dict[str, str] = {}
    if existing:
        for item in existing.split(","):
            if ":" in item:
                k, v = item.split(":", 1)
                existing_pairs[k.strip()] = v.strip()
    merged = {**desired, **existing_pairs}
    value = ",".join(f"{k}:{v}" for k, v in merged.items())
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = value
    logger.info("PYTORCH_CUDA_ALLOC_CONF=%s", value)
