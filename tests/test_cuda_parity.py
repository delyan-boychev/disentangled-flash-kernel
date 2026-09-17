"""Fast-path parity against the stock Transformers encoder. Requires CUDA + Triton.

Run on a supported GPU (SM 8.9 / 9.0):
    PYTHONPATH=torch-ext pytest tests/test_cuda_parity.py -v
"""

from __future__ import annotations

import types

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("triton")

from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model

from disentangled_flash import _encoder
from disentangled_flash.layers import DebertaV2Encoder as KernelEncoder

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

BASE = dict(
    hidden_size=768,
    num_hidden_layers=4,
    num_attention_heads=12,      # head dim 64
    intermediate_size=3072,
    max_position_embeddings=512,
    relative_attention=True,
    position_buckets=256,
    pos_att_type=["p2c", "c2p"],
    norm_rel_ebd="layer_norm",
    share_att_key=True,
    vocab_size=128100,
)

TOLERANCES = {
    torch.float16: dict(atol=2e-2, rtol=2e-2),
    torch.bfloat16: dict(atol=8e-2, rtol=8e-2),
    torch.float32: dict(atol=1e-5, rtol=1e-5),
}


def _model(dtype, **overrides):
    torch.manual_seed(0)
    model = DebertaV2Model(DebertaV2Config(**{**BASE, **overrides}))
    return model.eval().to(device="cuda", dtype=dtype)


def _kernelize(model):
    encoder = model.encoder
    encoder.forward = types.MethodType(KernelEncoder.forward, encoder)
    return model


def _inputs(batch, length, pad_to=None):
    torch.manual_seed(1)
    input_ids = torch.randint(0, BASE["vocab_size"], (batch, length), device="cuda")
    attention_mask = torch.ones(batch, length, dtype=torch.long, device="cuda")
    if pad_to is not None:
        attention_mask[:, pad_to:] = 0
    return input_ids, attention_mask


@cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch,length", [(1, 64), (2, 128), (4, 512)])
def test_matches_reference(dtype, batch, length):
    _encoder._STATES.clear()
    input_ids, attention_mask = _inputs(batch, length, pad_to=length // 2)

    baseline = _model(dtype)
    with torch.no_grad():
        expected = baseline(input_ids, attention_mask=attention_mask).last_hidden_state

    _kernelize(baseline)
    with torch.no_grad():
        actual = baseline(input_ids, attention_mask=attention_mask).last_hidden_state

    assert _encoder._STATES, "the fast path did not engage"
    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])


@cuda
def test_head_dim_128_is_supported():
    _encoder._STATES.clear()
    input_ids, attention_mask = _inputs(2, 128)
    model = _kernelize(_model(torch.float16, hidden_size=768, num_attention_heads=6))
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)
    assert _encoder._STATES, "head dim 128 should use the fast path"


@cuda
def test_unsupported_head_dim_falls_back():
    _encoder._STATES.clear()
    input_ids, attention_mask = _inputs(2, 128)
    # hidden 768 / 8 heads -> head dim 96, which the kernel does not support.
    model = _kernelize(_model(torch.float16, num_attention_heads=8))
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)
    assert not _encoder._STATES, "unsupported head dim must fall back"


@cuda
def test_fully_padded_rows_do_not_produce_nan():
    """Regression: a query whose first key tile is entirely masked.

    Reported on issue #48176 for heavily left-padded inputs, where the online
    softmax can hit -inf minus -inf.
    """
    _encoder._STATES.clear()
    batch, length = 2, 256
    input_ids = torch.randint(0, BASE["vocab_size"], (batch, length), device="cuda")
    attention_mask = torch.ones(batch, length, dtype=torch.long, device="cuda")
    attention_mask[:, : length - 8] = 0  # extreme left padding

    model = _kernelize(_model(torch.float16))
    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask).last_hidden_state
    assert torch.isfinite(out).all(), "non-finite values in the kernel output"


@cuda
@pytest.mark.parametrize("length", [64, 128, 384, 512])
def test_multiple_shapes_reuse_one_state(length):
    """Plans are cached per shape; the encoder state is built exactly once."""
    _encoder._STATES.clear()
    model = _kernelize(_model(torch.float16))
    for current in (64, 128, 384, 512):
        input_ids, attention_mask = _inputs(2, current)
        with torch.no_grad():
            model(input_ids, attention_mask=attention_mask)
    assert len(_encoder._STATES) == 1
    state = next(iter(_encoder._STATES.values()))
    assert len(state.plans) == 4


@cuda
def test_training_falls_back_on_cuda():
    _encoder._STATES.clear()
    input_ids, attention_mask = _inputs(2, 128)
    model = _kernelize(_model(torch.float32))
    model.train()
    out = model(input_ids, attention_mask=attention_mask)
    out.last_hidden_state.square().mean().backward()
    assert not _encoder._STATES, "training must not build kernel state"
