"""
pipeline/models/chronos/model_loader.py
=======================================
Singleton model manager and hardware device resolver for Amazon Chronos-2.

Provides:
1. Dynamic device resolution ('auto', 'cpu', 'cuda') with fallback diagnostics.
2. Singleton pipeline caching so the 120M-parameter (~480MB) model is loaded once
   in memory and shared across metric predictions, holdout evaluations, and runs.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from chronos import Chronos2Pipeline

from config.settings import (
    CHRONOS_DEVICE,
    CHRONOS_MODEL_ID,
    HF_TOKEN,
)

logger = logging.getLogger(__name__)

# Global cache for loaded pipeline instances: key is (model_id, device)
_PIPELINE_CACHE: dict[tuple[str, str], Chronos2Pipeline] = {}


def resolve_device(device_str: Optional[str] = None) -> str:
    """
    Resolve requested hardware device ('auto', 'cpu', 'cuda') to a valid execution target.

    - 'auto' (default): Uses CUDA if available; falls back to CPU.
    - 'cuda': Validates CUDA availability. If not available, logs a warning and falls back to CPU.
    - 'cpu': Forces CPU execution.

    Returns:
        str: Either 'cuda' or 'cpu'.
    """
    target = (device_str or CHRONOS_DEVICE or "auto").strip().lower()

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


def get_chronos_pipeline(
    model_id: str = CHRONOS_MODEL_ID,
    device: Optional[str] = None,
) -> Chronos2Pipeline:
    """
    Load or retrieve a cached instance of Amazon Chronos-2 pipeline.

    Args:
        model_id: HuggingFace model repo ID or local checkpoint path.
        device: 'auto', 'cpu', or 'cuda'.

    Returns:
        Chronos2Pipeline instance ready for inference.
    """
    resolved_dev = resolve_device(device)
    cache_key = (model_id, resolved_dev)

    if cache_key in _PIPELINE_CACHE:
        logger.debug("Reusing cached Chronos-2 instance for %s", cache_key)
        return _PIPELINE_CACHE[cache_key]

    logger.info(
        "Loading Amazon Chronos-2 ('%s') on target device='%s'...",
        model_id,
        resolved_dev,
    )

    token = HF_TOKEN or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    pipeline = Chronos2Pipeline.from_pretrained(
        pretrained_model_name_or_path=model_id,
        device_map=resolved_dev,
        token=token,
    )

    _PIPELINE_CACHE[cache_key] = pipeline
    logger.info(
        "✅ Chronos-2 loaded successfully on device='%s'.",
        resolved_dev,
    )
    return pipeline


def clear_chronos_cache() -> None:
    """Clear cached pipeline instances and trigger garbage collection."""
    global _PIPELINE_CACHE
    _PIPELINE_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Cleared Chronos-2 pipeline cache.")
