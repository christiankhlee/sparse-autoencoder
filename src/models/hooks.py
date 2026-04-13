"""
PyTorch forward-hook utilities for caching residual stream activations.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ActivationCache:
    """
    Stores residual-stream activations captured at a specified transformer layer.

    Internally this registers a forward hook on ``model.transformer.h[layer_idx]``
    and caches the first element of its output tuple (the hidden state tensor)
    for the most recent forward pass.

    Example::

        cache = ActivationCache()
        cache.register_hook(model, layer_idx=8)

        with torch.no_grad():
            model(input_ids)

        acts = cache.get_activations()  # shape: (batch, seq_len, d_model)
        cache.clear()
    """

    def __init__(self) -> None:
        self._activations: Optional[torch.Tensor] = None
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_hook(self, model: nn.Module, layer_idx: int) -> None:
        """
        Attach a forward hook to ``model.transformer.h[layer_idx]``.

        Any previously registered hooks are removed first so that calling
        this method again with a different layer is safe.

        Args:
            model:     A GPT-2 (or compatible) ``nn.Module``.
            layer_idx: Zero-based index of the transformer block to hook.
        """
        self.remove_hooks()

        try:
            target_layer: nn.Module = model.transformer.h[layer_idx]
        except (AttributeError, IndexError) as exc:
            raise ValueError(
                f"Could not access model.transformer.h[{layer_idx}]. "
                "Make sure the model is a GPT-2-style architecture."
            ) from exc

        hook = target_layer.register_forward_hook(self._hook_fn)
        self._hooks.append(hook)
        logger.debug("Registered activation hook on transformer.h[%d]", layer_idx)

    def get_activations(self) -> torch.Tensor:
        """
        Return the cached activation tensor from the most recent forward pass.

        Returns:
            Tensor of shape ``(batch_size, seq_len, d_model)``.

        Raises:
            RuntimeError: If no forward pass has been executed since the last
                          ``clear()`` call (or since construction).
        """
        if self._activations is None:
            raise RuntimeError(
                "No activations cached. Run a forward pass after calling "
                "register_hook()."
            )
        return self._activations

    def clear(self) -> None:
        """
        Discard the cached activation tensor.

        Call this between forward passes when you want to ensure stale data
        is never returned.
        """
        self._activations = None

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks from the model."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.debug("All activation hooks removed.")

    # ------------------------------------------------------------------
    # Internal hook callback
    # ------------------------------------------------------------------

    def _hook_fn(
        self,
        module: nn.Module,
        inputs: tuple,
        outputs: tuple | torch.Tensor,
    ) -> None:
        """
        PyTorch forward-hook callback.

        GPT-2 transformer blocks return a tuple whose first element is the
        hidden-state tensor of shape ``(batch, seq_len, d_model)``.  We
        detach and cache it.
        """
        # The block output is (hidden_state, ...) for GPT-2.
        if isinstance(outputs, tuple):
            hidden_state = outputs[0]
        else:
            hidden_state = outputs

        # Detach so we don't accidentally retain the computation graph.
        self._activations = hidden_state.detach()
