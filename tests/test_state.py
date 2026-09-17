"""The per-encoder state must share the host's weights, never copy them."""

from __future__ import annotations

import gc

import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model

from disentangled_flash import _encoder

TINY = dict(
    hidden_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    intermediate_size=128,
    max_position_embeddings=128,
    relative_attention=True,
    position_buckets=32,
    pos_att_type=["p2c", "c2p"],
    norm_rel_ebd="layer_norm",
    share_att_key=True,
    vocab_size=99,
)


@pytest.fixture
def model():
    torch.manual_seed(0)
    model = DebertaV2Model(DebertaV2Config(**TINY))
    model.eval()
    yield model
    _encoder._STATES.clear()
    _encoder._FINALIZERS.clear()


def test_state_shares_projection_modules(model):
    state = _encoder._build_state(model.encoder)
    assert len(state.attentions) == TINY["num_hidden_layers"]
    for shadow, layer in zip(state.attentions, model.encoder.layer):
        host = layer.attention.self
        # Identity, not equality: no second resident copy of the weights.
        assert shadow.query_proj is host.query_proj
        assert shadow.key_proj is host.key_proj
        assert shadow.value_proj is host.value_proj


def test_state_leaves_no_meta_tensors(model):
    """Shadow modules are built on meta and every parameter is replaced."""
    state = _encoder._build_state(model.encoder)
    for shadow in state.attentions:
        meta = [name for name, p in shadow.named_parameters() if p.is_meta]
        assert not meta, f"unreplaced meta parameters: {meta}"


def test_fused_qkv_preserves_checkpoint(model):
    """Packing Q/K/V rebinds storage but must not change state_dict content."""
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = _encoder._build_state(model.encoder)
    rel = model.encoder.get_rel_embedding()
    for attention in state.attentions:
        attention.prepare_for_inference(rel)

    after = model.state_dict()
    assert set(before) == set(after)
    for key, value in before.items():
        assert torch.equal(value, after[key]), f"{key} changed after fused QKV packing"


def test_fused_qkv_shares_storage_with_host(model):
    """After packing, the host's own parameters are views into the packed buffer."""
    state = _encoder._build_state(model.encoder)
    rel = model.encoder.get_rel_embedding()
    for attention in state.attentions:
        attention.prepare_for_inference(rel)

    for shadow, layer in zip(state.attentions, model.encoder.layer):
        host = layer.attention.self
        packed = shadow._cached_qkv_weight
        assert packed is not None
        for name in ("query_proj", "key_proj", "value_proj"):
            weight = getattr(host, name).weight
            assert weight.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr(), (
                f"{name}.weight is not a view into the packed QKV buffer"
            )


def test_state_is_released_with_the_encoder():
    """id() keys are only unique while alive; the finalizer must drop them."""
    _encoder._STATES.clear()
    torch.manual_seed(0)
    model = DebertaV2Model(DebertaV2Config(**TINY))
    model.eval()
    _encoder._state_for(model.encoder)
    assert len(_encoder._STATES) == 1

    del model
    gc.collect()
    assert not _encoder._STATES, "encoder state outlived the encoder"


def test_state_is_reused_across_calls(model):
    first = _encoder._state_for(model.encoder)
    second = _encoder._state_for(model.encoder)
    assert first is second
