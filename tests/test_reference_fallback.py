"""The fallback path must be the stock Transformers encoder, exactly.

Every input the Triton kernel cannot handle runs `_reference_forward`. Because
`kernelize` only replaces `forward`, that path calls the host module's own
`get_attention_mask` / `get_rel_pos` / `get_rel_embedding` and its untouched
layer modules, so it must be bit-identical to the un-kernelized encoder.

These tests run on CPU, where the kernel always falls back.
"""

from __future__ import annotations

import types

import pytest
import torch

pytest.importorskip("transformers")

from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model

from disentangled_flash.layers import DebertaV2Encoder as KernelEncoder

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


def _model(**overrides):
    torch.manual_seed(0)
    config = DebertaV2Config(**{**TINY, **overrides})
    model = DebertaV2Model(config)
    model.eval()
    return model


def _kernelize(model):
    """Swap only the encoder's forward, the way kernels.kernelize does."""
    encoder = model.encoder
    encoder.forward = types.MethodType(KernelEncoder.forward, encoder)
    return model


def _inputs(batch=2, length=16, vocab=99):
    torch.manual_seed(1)
    input_ids = torch.randint(0, vocab, (batch, length))
    attention_mask = torch.ones(batch, length, dtype=torch.long)
    attention_mask[0, length // 2 :] = 0  # real padding
    return input_ids, attention_mask


@pytest.mark.parametrize("output_hidden_states", [False, True])
@pytest.mark.parametrize("return_dict", [False, True])
def test_fallback_matches_reference(output_hidden_states, return_dict):
    input_ids, attention_mask = _inputs()
    baseline = _model()

    with torch.no_grad():
        expected = baseline(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    _kernelize(baseline)
    with torch.no_grad():
        actual = baseline(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    expected_hidden = expected[0] if not return_dict else expected.last_hidden_state
    actual_hidden = actual[0] if not return_dict else actual.last_hidden_state
    # Same code path, same tensors: this must be exact, not approximate.
    assert torch.equal(expected_hidden, actual_hidden)


def test_fallback_supports_output_attentions():
    """output_attentions is a fallback trigger; it must still work."""
    input_ids, attention_mask = _inputs()
    model = _kernelize(_model())
    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask, output_attentions=True)
    assert out.attentions is not None
    assert len(out.attentions) == TINY["num_hidden_layers"]


def test_fallback_under_training_and_grad():
    """Training must never reach the forward-only kernel."""
    input_ids, attention_mask = _inputs()
    model = _kernelize(_model())
    model.train()
    out = model(input_ids, attention_mask=attention_mask)
    loss = out.last_hidden_state.square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed through the fallback path"


def test_fallback_with_conv_layer():
    """SEW-D-style conv encoders are excluded from the fast path."""
    input_ids, attention_mask = _inputs()
    baseline = _model(conv_kernel_size=3)
    assert baseline.encoder.conv is not None
    with torch.no_grad():
        expected = baseline(input_ids, attention_mask=attention_mask).last_hidden_state
    _kernelize(baseline)
    with torch.no_grad():
        actual = baseline(input_ids, attention_mask=attention_mask).last_hidden_state
    assert torch.equal(expected, actual)


def test_no_state_is_built_on_the_fallback_path():
    """A CPU run must not allocate shadow attention modules."""
    from disentangled_flash import _encoder

    _encoder._STATES.clear()
    input_ids, attention_mask = _inputs()
    model = _kernelize(_model())
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)
    assert not _encoder._STATES


def test_checkpoint_keys_are_unchanged():
    """The adapter must never rename, add, or drop a state-dict key."""
    baseline = _model()
    before = dict(baseline.state_dict())
    _kernelize(baseline)
    input_ids, attention_mask = _inputs()
    with torch.no_grad():
        baseline(input_ids, attention_mask=attention_mask)
    after = dict(baseline.state_dict())
    assert set(before) == set(after)
    for key, value in before.items():
        assert torch.equal(value, after[key]), f"{key} changed value"
