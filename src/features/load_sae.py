"""
Utilities for loading pretrained Sparse Autoencoders from sae_lens.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_pretrained_sae(
    release: str,
    sae_id: str,
    device: str,
) -> tuple[Any, dict]:
    """
    Load a pretrained SAE from the ``sae_lens`` library.

    Args:
        release: SAE release name, e.g. ``"gpt2-small-res-jb"``.
        sae_id:  Specific SAE identifier within the release,
                 e.g. ``"blocks.8.hook_resid_post"``.
        device:  Target device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        ``(sae, cfg_dict)`` where *sae* is the loaded
        :class:`sae_lens.SAE` object and *cfg_dict* is the configuration
        dictionary that describes it.

    Raises:
        ImportError: If ``sae_lens`` is not installed.
        ValueError:  If the requested release / SAE ID cannot be found.
    """
    try:
        from sae_lens import SAE  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "sae_lens is required to load pretrained SAEs. "
            "Install it with: pip install sae-lens"
        ) from exc

    logger.info(
        "Loading pretrained SAE: release=%r  id=%r  device=%r",
        release,
        sae_id,
        device,
    )

    sae, cfg_dict, log_sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )

    # Move to the requested device (sae_lens may not honour the argument for
    # all releases).
    sae = sae.to(device)
    sae.eval()

    # Force-materialize any memory-mapped tensors into real RAM.
    # safetensors uses mmap by default; on macOS, evicted pages trigger SIGBUS
    # after repeated forward passes. Cloning pins all weight data in RAM.
    for param in sae.parameters():
        param.data = param.data.clone()

    n_features = cfg_dict.get("d_sae") or cfg_dict.get("n_features") or "unknown"
    logger.info(
        "SAE loaded successfully.  n_features=%s  log_sparsity=%s",
        n_features,
        f"{log_sparsity:.4f}" if log_sparsity is not None else "N/A",
    )

    return sae, cfg_dict
