"""
Feature extraction: run GPT-2 forward passes, collect residual-stream
activations at the target layer, encode them through the SAE, and return
the top-k active feature indices and values.
"""

from __future__ import annotations

import logging
from typing import Any, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.hooks import ActivationCache

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts top-k sparse features for a batch of prompts.

    The extractor:
    1. Runs the LM forward pass with a hook capturing the residual stream
       at ``layer_idx``.
    2. Takes the *last-token* hidden state and encodes it through the SAE.
    3. Returns the top-k feature indices and their activation values.

    Args:
        model:     GPT-2 (or compatible) causal LM.
        tokenizer: Corresponding tokenizer.
        sae:       Pretrained ``sae_lens.SAE`` object.
        layer_idx: Transformer block index to hook (0-based).
        device:    Torch device string.
        top_k:     Number of top features to retain per prompt.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        sae: Any,
        layer_idx: int,
        device: str,
        top_k: int = 20,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.layer_idx = layer_idx
        self.device = device
        self.top_k = top_k

        self._cache = ActivationCache()
        self._cache.register_hook(model, layer_idx)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_last_token_activation(self, prompt: str) -> torch.Tensor:
        """
        Run a forward pass and return the residual-stream vector for the
        final token at ``layer_idx``.

        Args:
            prompt: Input text.

        Returns:
            1-D tensor of shape ``(d_model,)``.
        """
        self._cache.clear()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        self.model(**inputs)

        # activations: (1, seq_len, d_model)
        activations = self._cache.get_activations()
        # Clone so we hold only (d_model,) and the full tensor can be freed now
        last_token_act = activations[0, -1, :].clone()
        self._cache.clear()
        return last_token_act

    @torch.no_grad()
    def get_feature_activations(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Pass a residual-stream vector through the SAE encoder.

        Args:
            activation: 1-D tensor of shape ``(d_model,)``.

        Returns:
            1-D tensor of shape ``(n_features,)`` — sparse feature activations.
        """
        # sae_lens expects (batch, d_model) or (seq_len, d_model).
        # We add a batch dimension then take [0] after encoding.
        act_2d = activation.unsqueeze(0)  # (1, d_model)
        feature_acts = self.sae.encode(act_2d)  # (1, n_features) or (n_features,)

        if feature_acts.dim() == 2:
            feature_acts = feature_acts[0]  # (n_features,)

        return feature_acts

    def get_top_k_features(
        self, activation: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Identify the top-k features by activation magnitude.

        Args:
            activation: 1-D tensor of shape ``(d_model,)`` — residual stream.

        Returns:
            ``(indices, values)`` — both 1-D NumPy arrays of length ``top_k``,
            sorted in descending order of activation value.
        """
        feature_acts = self.get_feature_activations(activation)
        k = min(self.top_k, feature_acts.numel())

        top_values, top_indices = torch.topk(feature_acts, k)

        return top_indices.cpu().numpy(), top_values.cpu().numpy()

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def extract_for_prompts(self, prompts: List[str]) -> List[dict]:
        """
        Extract sparse features for every prompt in the list.

        Args:
            prompts: List of input text strings.

        Returns:
            List of dicts, one per prompt, with keys:

            - ``"prompt"``         – original string
            - ``"top_k_indices"``  – list[int] of top-k feature indices
            - ``"top_k_values"``   – list[float] of corresponding activations
            - ``"full_activations"`` – list[float], full SAE feature vector
              (may be large — set to empty list if memory is a concern)
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.info("Prompt %d/%d: %r", i + 1, len(prompts), prompt[:60])
            try:
                last_act = self.get_last_token_activation(prompt)
                feature_acts = self.get_feature_activations(last_act)

                # Extract top-k directly from already-computed feature_acts
                # (avoids calling get_feature_activations a second time via get_top_k_features)
                k = min(self.top_k, feature_acts.numel())
                top_values_t, top_indices_t = torch.topk(feature_acts, k)
                top_indices = top_indices_t.cpu().numpy()
                top_values = top_values_t.cpu().numpy()
                del last_act, feature_acts, top_values_t, top_indices_t

                results.append(
                    {
                        "prompt": prompt,
                        "top_k_indices": top_indices.tolist(),
                        "top_k_values": top_values.tolist(),
                        "full_activations": [],
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to extract features for prompt %r: %s", prompt, exc)
                results.append(
                    {
                        "prompt": prompt,
                        "top_k_indices": [],
                        "top_k_values": [],
                        "full_activations": [],
                        "error": str(exc),
                    }
                )

        return results

    # ------------------------------------------------------------------
    # Next-token prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_next_token_prediction(self, prompt: str) -> str:
        """
        Return the model's top-1 next-token prediction as a decoded string.

        Args:
            prompt: Input text.

        Returns:
            Decoded next token (may include a leading space, stripped).
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)
        next_token_id = logits[0, -1, :].argmax(dim=-1).item()
        next_token = self.tokenizer.decode(next_token_id)
        return next_token

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        try:
            self._cache.remove_hooks()
        except Exception:  # noqa: BLE001
            pass
