#!/usr/bin/env python3
"""Verify a built kernel through the real `kernelize` path.

The pytest suite binds `forward` with `types.MethodType`, which does not
exercise how `kernels` actually resolves and installs a layer out of a built
package. This script does, and is the gate to pass before uploading to the Hub.

It works against released Transformers: `replace_kernel_forward_from_hub` makes
the stock `DebertaV2Encoder` extensible at runtime, exactly as the
`@use_kernel_forward_from_hub` decorator does in the pending Transformers PR.

Usage:
    python scripts/verify_build.py                      # defaults to ./build
    python scripts/verify_build.py --build-dir build --model microsoft/deberta-v3-base
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from kernels import (
    LocalLayerRepository,
    Mode,
    kernelize,
    replace_kernel_forward_from_hub,
    use_kernel_mapping,
)
from transformers import AutoConfig, AutoModel
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder

LAYER = "DebertaV2Encoder"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="  [kernels] %(message)s")

    if not torch.cuda.is_available():
        print("FAIL: CUDA is required", file=sys.stderr)
        return 1
    build_dir = args.build_dir.resolve()
    if not build_dir.is_dir():
        print(f"FAIL: no build directory at {build_dir}", file=sys.stderr)
        return 1
    print(f"build   : {build_dir}")
    print(f"variants: {sorted(p.name for p in build_dir.iterdir() if p.is_dir())}")

    dtype = getattr(torch, args.dtype)
    config = AutoConfig.from_pretrained(args.model)
    torch.manual_seed(0)
    model = AutoModel.from_config(config).eval().to("cuda", dtype)

    input_ids = torch.randint(0, config.vocab_size, (args.batch, args.length), device="cuda")
    attention_mask = torch.ones(args.batch, args.length, dtype=torch.long, device="cuda")
    attention_mask[:, args.length // 2 :] = 0

    with torch.no_grad():
        expected = model(input_ids, attention_mask=attention_mask).last_hidden_state

    # Stock Transformers has no decorator yet; this is what the PR adds.
    replace_kernel_forward_from_hub(DebertaV2Encoder, LAYER)
    encoder = model.encoder if hasattr(model, "encoder") else model.deberta.encoder
    before = encoder.forward

    mapping = {
        LAYER: {
            "cuda": LocalLayerRepository(repo_path=build_dir, layer_name=LAYER),
        }
    }
    # inherit_mapping=False: only this kernel, nothing from the default mapping.
    # use_fallback=False: raise instead of silently leaving the layer unkernelized.
    with use_kernel_mapping(mapping, inherit_mapping=False):
        kernelize(model, mode=Mode.INFERENCE, device="cuda", use_fallback=False)

    if encoder.forward == before:
        print("\nFAIL: kernelize did not replace DebertaV2Encoder.forward", file=sys.stderr)
        return 1
    print(f"\nforward swapped: {before} -> {encoder.forward}")

    with torch.no_grad():
        actual = model(input_ids, attention_mask=attention_mask).last_hidden_state

    if not torch.isfinite(actual).all():
        print("FAIL: kernel produced non-finite values", file=sys.stderr)
        return 1

    difference = (actual.float() - expected.float()).abs()
    tolerance = {"float16": 2e-2, "bfloat16": 8e-2, "float32": 1e-4}[args.dtype]
    print(f"max abs error  : {difference.max().item():.3e}")
    print(f"mean abs error : {difference.mean().item():.3e}")

    if difference.max().item() > tolerance:
        print(f"FAIL: exceeds tolerance {tolerance:.1e}", file=sys.stderr)
        return 1

    print("\nPASS: the built kernel loads, installs and matches the reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
