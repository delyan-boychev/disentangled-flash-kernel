"""Hub kernel layers.

``kernelize()`` replaces the ``forward`` of a Transformers module with the
``forward`` defined here. Per the Hub layer contract these classes must be
pure: they define no constructor, no class variables other than the two
capability flags, and no method other than ``forward``. Every attribute they
read comes from the adopting Transformers module and is declared below as a
type annotation only.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ._encoder import encoder_forward

__all__ = ["DebertaV2Encoder"]


class DebertaV2Encoder(nn.Module):
    """Fused disentangled attention for ``DebertaV2Encoder``.

    Replaces the dense ``[B, H, L, L]`` attention path of DeBERTa-v2/v3 with a
    tiled, streaming Triton kernel that never materializes the attention
    matrix. Inference only; unsupported inputs and modes transparently run the
    stock Transformers encoder loop.
    """

    # Forward-only kernel: kernelize() keeps the reference forward for training.
    has_backward: bool = False
    # Shape-dependent plan preparation is not capturable; keep eager.
    can_torch_compile: bool = False

    # Supplied by the adopting transformers.DebertaV2Encoder.
    layer: nn.ModuleList
    rel_embeddings: nn.Embedding
    relative_attention: bool
    conv: Any

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> Any:
        return encoder_forward(
            self,
            hidden_states,
            attention_mask,
            output_hidden_states,
            output_attentions,
            query_states,
            relative_pos,
            return_dict,
        )
