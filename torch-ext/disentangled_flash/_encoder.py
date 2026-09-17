"""Stateless ``DebertaV2Encoder`` adapter.

Hub kernel layers must be pure: no constructor, no class variables beyond the
capability flags, and no method other than ``forward``. All of the machinery
that the standalone ``disentangled-flash`` package keeps on its encoder module
lives here instead, in a module-level cache keyed by the identity of the host
Transformers encoder.

The adapter never replaces submodules and never copies weights. It builds
shadow attention modules that *share* the host's ``nn.Linear`` objects by
reference, so ``state_dict()`` keys and values are untouched and there is no
second resident copy of the projection weights.

Anything the kernel does not support falls back to the stock Transformers
encoder loop. That boundary is what keeps the kernel safe across upstream
refactors.
"""

from __future__ import annotations

import logging
import weakref
from collections import OrderedDict
from typing import Any

import torch

from ._reference import BaseModelOutput, DebertaAttentionConfig
from .kernel import TritonInferenceDisentangledSelfAttention, triton
from .position import SharedPositionPlanCache
from .tuning import KernelTuningOptions, ProfileRegistry

logger = logging.getLogger(__name__)

MAX_SEQUENCE_LENGTH = 8192
SUPPORTED_HEAD_DIMS = frozenset({32, 64, 128})
SUPPORTED_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})

# Attributes the adapter reads off the host attention module. Checked before
# every fast-path entry so an upstream rename degrades to the eager loop
# instead of raising.
REQUIRED_ATTENTION_ATTRS = (
    "query_proj",
    "key_proj",
    "value_proj",
    "num_attention_heads",
    "attention_head_size",
    "pos_att_type",
    "share_att_key",
    "relative_attention",
    "position_buckets",
    "max_relative_positions",
)

# Distinct sequence lengths whose position plans are retained per encoder.
MAX_CACHED_SHAPES = 16

_STATES: dict[int, "_EncoderState"] = {}
_FINALIZERS: dict[int, weakref.finalize] = {}
_WARNED: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _WARNED:
        _WARNED.add(message)
        logger.warning("disentangled-flash: %s; using the reference encoder", message)


class _EncoderState:
    """Per-encoder plan cache. Lives in ``_STATES``, never on the layer class."""

    __slots__ = ("attentions", "plan_cache", "plans")

    def __init__(
        self,
        attentions: tuple[TritonInferenceDisentangledSelfAttention, ...],
        plan_cache: SharedPositionPlanCache,
    ) -> None:
        self.attentions = attentions
        self.plan_cache = plan_cache
        self.plans: OrderedDict[tuple[int, str], tuple[Any, ...]] = OrderedDict()


def _normalize_pos_att_type(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip().lower() for part in value.split("|") if part.strip())
    return tuple(str(part).strip().lower() for part in (value or ()))


def _attention_config(attention: Any) -> DebertaAttentionConfig:
    """Rebuild the config subset the vendored attention needs from a live module."""
    max_relative_positions = int(attention.max_relative_positions)
    return DebertaAttentionConfig(
        hidden_size=attention.query_proj.in_features,
        num_attention_heads=int(attention.num_attention_heads),
        attention_head_size=int(attention.attention_head_size),
        # Inference only: dropout is identity in eval() regardless, but pinning
        # the probabilities to zero keeps the shadow modules unambiguous.
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        relative_attention=bool(attention.relative_attention),
        max_relative_positions=max_relative_positions,
        max_position_embeddings=max_relative_positions,
        position_buckets=int(attention.position_buckets),
        share_att_key=bool(attention.share_att_key),
        pos_att_type=_normalize_pos_att_type(attention.pos_att_type),
    )


def _share_projections(shadow: Any, host: Any) -> None:
    """Point the shadow module at the host's projection modules by reference.

    Assigning the whole ``nn.Linear`` (not a weight copy) means the packed QKV
    buffer built by ``prepare_for_inference`` rebinds the *host's* parameters as
    storage views, exactly as the standalone package does. Checkpoint keys and
    values are preserved and no weight is duplicated.
    """
    for name in ("query_proj", "key_proj", "value_proj", "pos_key_proj", "pos_query_proj"):
        source = getattr(host, name, None)
        if source is not None:
            setattr(shadow, name, source)


def _build_state(encoder: Any) -> _EncoderState:
    first = encoder.layer[0].attention.self
    config = _attention_config(first)
    pos_ebd_size = config.position_buckets if config.position_buckets > 0 else config.max_relative_positions
    uses_position_bias = config.relative_attention and bool(
        {"c2p", "p2c"}.intersection(config.pos_att_type)
    )
    plan_cache = SharedPositionPlanCache(
        position_buckets=config.position_buckets,
        max_relative_positions=config.max_relative_positions,
        position_embedding_size=pos_ebd_size,
        uses_position_bias=uses_position_bias,
    )

    # Zero-config tuning: a bundled profile when the stack matches, bounded
    # autotuning otherwise. One registry is shared by every layer.
    tuning = KernelTuningOptions()
    registry = ProfileRegistry.from_options(tuning)

    attentions = []
    for layer in encoder.layer:
        host = layer.attention.self
        # Build on the meta device so the shadow's throwaway nn.Linear weights
        # are never allocated; every one of them is replaced below.
        with torch.device("meta"):
            shadow = TritonInferenceDisentangledSelfAttention(
                _attention_config(host),
                position_plan_cache=plan_cache,
                tuning=tuning,
                profile_registry=registry,
            )
        _share_projections(shadow, host)
        shadow.eval()
        attentions.append(shadow)

    return _EncoderState(tuple(attentions), plan_cache)


def _drop_state(key: int) -> None:
    _STATES.pop(key, None)
    _FINALIZERS.pop(key, None)


def _state_for(encoder: Any) -> _EncoderState:
    key = id(encoder)
    state = _STATES.get(key)
    if state is None:
        state = _build_state(encoder)
        _STATES[key] = state
        # id() is only unique while the object is alive; drop the entry with it.
        _FINALIZERS[key] = weakref.finalize(encoder, _drop_state, key)
    return state


def _plans_for(
    state: _EncoderState,
    encoder: Any,
    sequence_length: int,
    device: torch.device,
) -> tuple[Any, ...]:
    key = (sequence_length, str(device))
    plans = state.plans.get(key)
    if plans is not None:
        state.plans.move_to_end(key)
        return plans

    rel_embeddings = encoder.get_rel_embedding()
    for attention in state.attentions:
        # Rebuilds the fused QKV pack and the full projected position tables.
        # Only ever runs on a cold sequence length.
        attention.prepare_for_inference(rel_embeddings)
    plans = tuple(
        attention.prepare_shape(sequence_length, device) for attention in state.attentions
    )
    for attention in state.attentions:
        # The compact per-shape tables are built; the full tables were workspace.
        attention.release_position_projection_workspace()

    state.plans[key] = plans
    while len(state.plans) > MAX_CACHED_SHAPES:
        state.plans.popitem(last=False)
    return plans


def _unsupported_reason(
    encoder: Any,
    hidden_states: torch.Tensor,
    attention_mask: Any,
    output_attentions: bool,
    query_states: Any,
    relative_pos: Any,
) -> str | None:
    """Return why the fast path cannot run, or ``None`` when it can."""
    if encoder.training or torch.is_grad_enabled():
        return "training or autograd is enabled (this kernel is inference-only)"
    if output_attentions:
        return "output_attentions=True is not supported"
    if query_states is not None:
        return "query_states (z_steps) is not supported"
    if relative_pos is not None:
        return "a caller-supplied relative_pos is not supported"
    if getattr(encoder, "conv", None) is not None:
        return "encoders with a convolution layer are not supported"
    if not getattr(encoder, "relative_attention", False):
        return "relative attention is disabled"
    if triton is None:
        return "Triton is not installed"
    if hidden_states.device.type != "cuda":
        return f"device '{hidden_states.device.type}' is not CUDA"
    if hidden_states.dtype not in SUPPORTED_DTYPES:
        return f"dtype {hidden_states.dtype} is not supported"
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.dim() != 2:
        return "attention_mask is not a 2-D padding mask"
    if tuple(attention_mask.shape[:2]) != tuple(hidden_states.shape[:2]):
        return "attention_mask does not match hidden_states [B, L]"

    sequence_length = hidden_states.size(1)
    if sequence_length > MAX_SEQUENCE_LENGTH:
        return f"sequence length {sequence_length} exceeds the supported maximum {MAX_SEQUENCE_LENGTH}"

    layers = getattr(encoder, "layer", None)
    if not layers:
        return "the encoder exposes no layer list"
    attention = getattr(getattr(layers[0], "attention", None), "self", None)
    if attention is None:
        return "unrecognised attention module layout"
    for name in REQUIRED_ATTENTION_ATTRS:
        if not hasattr(attention, name):
            return f"the attention module has no attribute '{name}'"
    if int(attention.attention_head_size) not in SUPPORTED_HEAD_DIMS:
        return f"attention head size {attention.attention_head_size} is not one of {sorted(SUPPORTED_HEAD_DIMS)}"
    if not {"c2p", "p2c"}.intersection(_normalize_pos_att_type(attention.pos_att_type)):
        return "neither c2p nor p2c disentangled attention is enabled"
    if not hasattr(encoder, "get_rel_embedding"):
        return "the encoder has no get_rel_embedding()"
    return None


def _reference_forward(
    encoder: Any,
    hidden_states: torch.Tensor,
    attention_mask: Any,
    output_hidden_states: bool,
    output_attentions: bool,
    query_states: Any,
    relative_pos: Any,
    return_dict: bool,
) -> Any:
    """The stock ``DebertaV2Encoder.forward`` body, run against the host module.

    ``kernelize`` only replaces ``forward``, so ``get_attention_mask``,
    ``get_rel_pos`` and ``get_rel_embedding`` are still the upstream
    implementations and the layer modules are untouched. This is the genuine
    reference path, not an approximation of it.
    """
    if attention_mask.dim() <= 2:
        input_mask = attention_mask
    else:
        input_mask = attention_mask.sum(-2) > 0
    attention_mask = encoder.get_attention_mask(attention_mask)
    relative_pos = encoder.get_rel_pos(hidden_states, query_states, relative_pos)

    all_hidden_states = (hidden_states,) if output_hidden_states else None
    all_attentions = () if output_attentions else None

    next_kv = hidden_states
    rel_embeddings = encoder.get_rel_embedding()
    output_states = hidden_states
    for i, layer_module in enumerate(encoder.layer):
        output_states, attn_weights = layer_module(
            next_kv,
            attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            output_attentions=output_attentions,
        )
        if output_attentions:
            all_attentions = all_attentions + (attn_weights,)
        if i == 0 and getattr(encoder, "conv", None) is not None:
            output_states = encoder.conv(hidden_states, output_states, input_mask)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)
        if query_states is not None:
            query_states = output_states
        else:
            next_kv = output_states

    if not return_dict:
        return tuple(v for v in (output_states, all_hidden_states, all_attentions) if v is not None)
    return BaseModelOutput(
        last_hidden_state=output_states,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
    )


@torch.no_grad()
def _fast_forward(
    encoder: Any,
    state: _EncoderState,
    plans: tuple[Any, ...],
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    output_hidden_states: bool,
    return_dict: bool,
) -> Any:
    """Streaming path: no dense mask and no dense relative-position tensor."""
    mask = attention_mask
    if mask.dtype != torch.bool or not mask.is_contiguous():
        mask = mask.bool().contiguous()

    all_hidden_states = (hidden_states,) if output_hidden_states else None
    next_kv = hidden_states
    output_states = hidden_states

    for layer, attention, plan in zip(encoder.layer, state.attentions, plans):
        self_output, _ = attention.forward_prepared(next_kv, mask, plan)
        attention_output = layer.attention.output(self_output, next_kv)
        intermediate_output = layer.intermediate(attention_output)
        output_states = layer.output(intermediate_output, attention_output)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)
        next_kv = output_states

    if not return_dict:
        return tuple(v for v in (output_states, all_hidden_states) if v is not None)
    return BaseModelOutput(
        last_hidden_state=output_states,
        hidden_states=all_hidden_states,
        attentions=None,
    )


def encoder_forward(
    encoder: Any,
    hidden_states: torch.Tensor,
    attention_mask: Any,
    output_hidden_states: bool = True,
    output_attentions: bool = False,
    query_states: Any = None,
    relative_pos: Any = None,
    return_dict: bool = True,
) -> Any:
    """Entry point for ``layers.DebertaV2Encoder.forward``."""
    reason = _unsupported_reason(
        encoder, hidden_states, attention_mask, output_attentions, query_states, relative_pos
    )
    if reason is not None:
        _warn_once(reason)
        return _reference_forward(
            encoder,
            hidden_states,
            attention_mask,
            output_hidden_states,
            output_attentions,
            query_states,
            relative_pos,
            return_dict,
        )

    state = _state_for(encoder)
    plans = _plans_for(state, encoder, hidden_states.size(1), hidden_states.device)
    return _fast_forward(
        encoder, state, plans, hidden_states, attention_mask, output_hidden_states, return_dict
    )
