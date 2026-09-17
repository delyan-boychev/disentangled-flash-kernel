# disentangled-flash-kernel

Hub-kernel packaging of [DisentangledFlash](https://github.com/delyan-boychev/disentangled-flash) —
fast exact DeBERTa-v2/v3 disentangled attention in Triton.

Published to the Hub as [`delyan-boychev/disentangled-flash`](https://huggingface.co/delyan-boychev/disentangled-flash)
and loaded by Transformers through the `kernels` library. See [`CARD.md`](CARD.md) for the
user-facing documentation.

## Why this is a separate repository

Hub kernels may only import from the standard library, torch, or the kernel itself, so a
kernel cannot depend on the `disentangled-flash` PyPI package. The implementation is
vendored here from a pinned upstream tag; the standalone repository is unchanged and
remains the home for offline tuning, packed inference and compilation buckets.

## Layout

```
build.toml                          # [torch-noarch] pure-Triton, no compile matrix
flake.nix                           # kernel-builder entry point
torch-ext/disentangled_flash/
  layers.py                         # the pure Hub layer (kernelize swaps this forward)
  _encoder.py                       # stateless adapter + reference fallback
  _reference.py _torch.py           # vendored, verbatim
  kernel.py position.py packed.py tuning.py _validation.py
  profiles/*.json                   # bundled tuning profiles (shipped via pyext)
  _vendored.py                      # generated upstream provenance stamp
scripts/vendor.py                   # re-vendor from a pinned upstream ref
tests/
```

## Re-vendoring

```bash
python scripts/vendor.py --source ../disentangled-flash --ref v0.2.0
```

The script refuses to vendor a module that imports anything outside stdlib, torch or
triton, which is what keeps the kernel Hub-compliant.

## Tests

```bash
PYTHONPATH=torch-ext pytest tests/
```

`test_layer_contract.py` enforces the Hub layer purity rules and checks the `forward`
signature against the installed Transformers. `test_reference_fallback.py` and
`test_state.py` run anywhere. `test_cuda_parity.py` needs CUDA + Triton.

## Building

```bash
nix build .#redistributable        # or: kernel-builder build .
```

Produces `build/torch-cuda/`. Verify before uploading:

```python
from kernels import LocalLayerRepository, Mode, kernelize, use_kernel_mapping

with use_kernel_mapping(
    {"DebertaV2Encoder": {"cuda": LocalLayerRepository(
        repo_path="build", package_name="disentangled_flash",
        layer_name="DebertaV2Encoder")}},
    inherit_mapping=False,
):
    kernelize(model, mode=Mode.INFERENCE)
```

## License

Apache-2.0. See [`NOTICE`](NOTICE) for attribution of the vendored Transformers
reference code.
