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
        name for name, value in vars(layer).items() if callable(value) and not name.startswith("__")
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


def test_no_absolute_self_references_in_vendored_code():
    """Vendored modules must never refer to this package by its absolute name.

    `kernels` loads a Hub kernel under a uniquely generated module name, so an
    absolute self-reference either raises ModuleNotFoundError or silently
    resolves to a *different* installation of the package. The requirements put
    it as: "All Python code imports from the kernel itself must be relative."

    `scripts/vendor.py` catches absolute *imports*; this catches the other
    spellings, e.g. `resources.files("disentangled_flash.profiles")`.

    Only module-resolution contexts are flagged. Filesystem paths that happen to
    contain the project name are fine -- `default_user_profile_directory()`
    deliberately points at ~/.cache/disentangled_flash/profiles, which is shared
    with the standalone package on purpose.
    """
    import re
    from pathlib import Path

    package_root = Path(__import__("disentangled_flash").__file__).parent
    pattern = re.compile(
        r"""(?:resources\.files|import_module|__import__|sys\.modules\s*\[)"""
        r"""\s*\(?\s*["']disentangled_flash["'.]"""
    )

    offenders = []
    for source in sorted(package_root.rglob("*.py")):
        for number, line in enumerate(source.read_text().splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{source.relative_to(package_root)}:{number}: {line.strip()}")

    assert not offenders, "absolute self-reference(s) found:\n  " + "\n  ".join(offenders)


def test_no_python_310_only_constructs():
    """The kernel must stay importable on Python 3.9.

    "Python code must be compatible with Python 3.9 and later" -- Kernel Hub
    requirements. torch 2.8 is a supported build variant and still runs on 3.9,
    so this is reachable, not theoretical.

    Catches the realistic 3.10-only spellings rather than attempting a full
    version analysis. Vendored code is included: a re-vendor from upstream is
    how such a construct would return.
    """
    import ast
    from pathlib import Path

    package_root = Path(__import__("disentangled_flash").__file__).parent

    # (callable or attribute name, why it is 3.10+)
    banned_names = {
        "pairwise": "itertools.pairwise is 3.10+",
        "anext": "anext() is 3.10+",
        "aiter": "aiter() is 3.10+",
        "bit_count": "int.bit_count() is 3.10+",
    }

    offenders = []
    for source in sorted(package_root.rglob("*.py")):
        tree = ast.parse(source.read_text(), filename=str(source))
        relative = source.relative_to(package_root)
        for node in ast.walk(tree):
            # `match` statements are 3.10 syntax.
            if isinstance(node, ast.Match):
                offenders.append(f"{relative}:{node.lineno}: match statement is 3.10+")
            # zip(..., strict=...) is 3.10+.
            if isinstance(node, ast.Call):
                function = node.func
                name = getattr(function, "id", None) or getattr(function, "attr", None)
                if name == "zip" and any(k.arg == "strict" for k in node.keywords):
                    offenders.append(f"{relative}:{node.lineno}: zip(strict=) is 3.10+")
                if name in banned_names:
                    offenders.append(f"{relative}:{node.lineno}: {banned_names[name]}")
            # Bare `from itertools import pairwise`.
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in banned_names:
                        offenders.append(f"{relative}:{node.lineno}: {banned_names[alias.name]}")

    assert not offenders, "Python 3.10+ construct(s) found:\n  " + "\n  ".join(offenders)
