"""
pipeline/models/timesfm/model_loader.py
=======================================
Singleton model manager and hardware device resolver for Google TimesFM 3.0.

Provides:
1. Dynamic device resolution ('auto', 'cpu', 'cuda') with fallback diagnostics.
2. Singleton model caching so the 330M-parameter (~1.3GB) model is loaded once
   in memory and shared across metric predictions, holdout evaluations, and runs.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import timesfm

from config.settings import (
    HF_TOKEN,
    TIMESFM_DEVICE,
    TIMESFM_MODEL_ID,
    TIMESFM_PER_CORE_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

# Global cache for loaded model instances: key is (model_id, device, batch_size)
_MODEL_CACHE: dict[tuple[str, str, int], timesfm.TimesFM3Forecaster] = {}


def resolve_device(device_str: Optional[str] = None) -> str:
    """
    Resolve requested hardware device ('auto', 'cpu', 'cuda') to a valid execution target.

    - 'auto' (default): Uses CUDA if available; falls back to CPU.
    - 'cuda': Validates CUDA availability. If not available, logs a warning and falls back to CPU.
    - 'cpu': Forces CPU execution.

    Returns:
        str: Either 'cuda' or 'cpu'.
    """
    target = (device_str or TIMESFM_DEVICE or "auto").strip().lower()

    if target == "auto":
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("Auto-detected GPU: %s. Using device='cuda'.", device_name)
            return "cuda"
        logger.info("No GPU detected. Using device='cpu'.")
        return "cpu"

    if target in ("cuda", "gpu"):
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("Using GPU: %s ('cuda').", device_name)
            return "cuda"
        logger.warning(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Falling back to CPU execution."
        )
        return "cpu"

    return "cpu"


def get_timesfm_model(
    model_id: str = TIMESFM_MODEL_ID,
    device: Optional[str] = None,
    per_core_batch_size: int = TIMESFM_PER_CORE_BATCH_SIZE,
) -> timesfm.TimesFM3Forecaster:
    """
    Load or retrieve a cached instance of Google TimesFM 3.0.

    Args:
        model_id: HuggingFace model repo ID or local checkpoint path.
        device: 'auto', 'cpu', or 'cuda'.
        per_core_batch_size: Batch size for parallel time-series inference.

    Returns:
        timesfm.TimesFM3Forecaster instance ready for inference.
    """
    resolved_dev = resolve_device(device)
    cache_key = (model_id, resolved_dev, per_core_batch_size)

    if cache_key in _MODEL_CACHE:
        logger.debug("Reusing cached TimesFM 3.0 instance for %s", cache_key)
        return _MODEL_CACHE[cache_key]

    logger.info(
        "Loading Google TimesFM 3.0 ('%s') on target device='%s' (batch_size=%d)...",
        model_id,
        resolved_dev,
        per_core_batch_size,
    )

    token = HF_TOKEN or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    forecaster = timesfm.TimesFM3Forecaster.from_pretrained(
        pretrained_model_name_or_path=model_id,
        device=resolved_dev,
        per_core_batch_size=per_core_batch_size,
        token=token,
    )

    _MODEL_CACHE[cache_key] = forecaster
    logger.info(
        "✅ TimesFM 3.0 loaded successfully on device='%s'.",
        resolved_dev,
    )
    return forecaster


def clear_timesfm_cache() -> None:
    """Clear cached model instances and trigger garbage collection."""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleared TimesFM model cache.")
