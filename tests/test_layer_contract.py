"""Enforce the Hub layer contract.

https://huggingface.co/docs/kernels/main/en/kernel-requirements#writing-layers

  - layers subclass torch.nn.Module
  - layers are pure: no constructor, no class variables
  - no method other than forward
  - the forward signature matches the layer it extends

The two permitted exceptions to the no-class-variables rule are
``has_backward`` and ``can_torch_compile``.
"""

from __future__ import annotations

import inspect

import pytest
import torch.nn as nn

from disentangled_flash import layers

ALLOWED_CLASS_VARS = {"has_backward", "can_torch_compile"}

LAYER_CLASSES = [getattr(layers, name) for name in layers.__all__]


@pytest.mark.parametrize("layer", LAYER_CLASSES, ids=lambda c: c.__name__)
def test_is_module_subclass(layer):
    assert issubclass(layer, nn.Module)


@pytest.mark.parametrize("layer", LAYER_CLASSES, ids=lambda c: c.__name__)
def test_defines_no_constructor(layer):
    assert "__init__" not in vars(layer), "Hub layers must not define a constructor"


@pytest.mark.parametrize("layer", LAYER_CLASSES, ids=lambda c: c.__name__)
def test_defines_only_forward(layer):
    methods = {
        name
        for name, value in vars(layer).items()
        if callable(value) and not name.startswith("__")
    }
    assert methods == {"forward"}, f"unexpected methods: {sorted(methods - {'forward'})}"


@pytest.mark.parametrize("layer", LAYER_CLASSES, ids=lambda c: c.__name__)
def test_defines_no_class_variables(layer):
    class_vars = {
        name
        for name, value in vars(layer).items()
        if not name.startswith("__") and not callable(value)
    }
    unexpected = class_vars - ALLOWED_CLASS_VARS
    assert not unexpected, f"Hub layers must be stateless; found {sorted(unexpected)}"


@pytest.mark.parametrize("layer", LAYER_CLASSES, ids=lambda c: c.__name__)
def test_annotations_carry_no_values(layer):
    """Host-supplied attributes must be annotations only, never assignments."""
    for name in getattr(layer, "__annotations__", {}):
        if name in ALLOWED_CLASS_VARS:
            continue
        assert name not in vars(layer), f"{name} is annotated and assigned; annotate only"


def test_forward_signature_matches_transformers():
    """The layer's forward must be signature-compatible with the one it replaces."""
    transformers = pytest.importorskip("transformers")
    from transformers.models.deberta_v2 import modeling_deberta_v2

    upstream = inspect.signature(modeling_deberta_v2.DebertaV2Encoder.forward)
    ours = inspect.signature(layers.DebertaV2Encoder.forward)

    upstream_params = list(upstream.parameters)
    our_params = list(ours.parameters)
    assert our_params == upstream_params, (
        f"signature drift vs transformers {transformers.__version__}:\n"
        f"  upstream: {upstream_params}\n"
        f"  ours:     {our_params}"
    )

    for name, upstream_param in upstream.parameters.items():
        if upstream_param.default is not inspect.Parameter.empty:
            assert ours.parameters[name].default == upstream_param.default, (
                f"default for {name!r} differs from upstream"
            )


def test_package_exports_layers():
    """`kernels` discovers layers through the package's __init__."""
    import disentangled_flash

    assert "layers" in disentangled_flash.__all__
    assert disentangled_flash.layers is layers
