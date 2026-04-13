"""
Model and tokenizer loading utilities.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    device: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a causal language model and its tokenizer from Hugging Face.

    The function resolves device availability automatically: if *device* is
    ``"cuda"`` but CUDA is not available it falls back to CPU with a warning.

    Args:
        model_name: Hugging Face model identifier, e.g. ``"gpt2"``.
        device:     Target device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        ``(model, tokenizer)`` both placed / configured for *device*.
    """
    # ---- device resolution -------------------------------------------
    resolved_device = _resolve_device(device)

    # ---- tokenizer ---------------------------------------------------
    logger.info("Loading tokenizer for '%s' ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 has no pad token by default; use the EOS token instead.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("Set pad_token = eos_token (%r)", tokenizer.eos_token)

    # ---- model -------------------------------------------------------
    logger.info("Loading model '%s' onto device '%s' ...", model_name, resolved_device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # stay in float32 for interpretability work
    )
    model = model.to(resolved_device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded: %d parameters (%.1f M), device=%s",
        n_params,
        n_params / 1e6,
        resolved_device,
    )

    return model, tokenizer


def _resolve_device(requested: str) -> str:
    """
    Validate and resolve the requested compute device.

    Args:
        requested: ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.

    Returns:
        Resolved device string safe to pass to ``tensor.to()``.
    """
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA was requested but is not available. Falling back to CPU."
            )
            return "cpu"
        return requested

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS was requested but is not available. Falling back to CPU.")
            return "cpu"
        return requested

    return requested  # "cpu" or anything else
