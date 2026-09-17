---
library_name: kernels
license: apache-2.0
tags:
  - kernel
  - triton
  - deberta
  - attention
---

# DisentangledFlash

Fast exact DeBERTa-v2/v3 disentangled attention in Triton, packaged as a Hub kernel.

DeBERTa's content-to-position (C2P) and position-to-content (P2C) terms prevent it from
using SDPA or FlashAttention, so Transformers materializes the full `[B, H, L, L]`
attention matrix. This kernel evaluates the same formulation with a tiled, streaming
Triton kernel that never materializes it.

Source: <https://github.com/delyan-boychev/disentangled-flash>

## Usage

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v2-xlarge-mnli",
    use_kernels=True,
    device_map="cuda",
)
```

Requires `pip install -U "transformers[kernels]"`. The kernel engages automatically on
the GPUs listed below; every other device keeps the stock PyTorch path.

To select it explicitly, or on a GPU outside the default range:

```python
from transformers import AutoModelForSequenceClassification, KernelConfig

kernel_config = KernelConfig(
    kernel_mapping={
        "DebertaV2Encoder": (
            "delyanboychev/disentangled-flash:DebertaV2Encoder",
            {"version": 0, "trust_remote_code": True},
        ),
    }
)
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    use_kernels=True,
    kernel_config=kernel_config,
    device_map="cuda",
)
```

## Scope

| | |
|---|---|
| Layer | `DebertaV2Encoder` (DeBERTa-v2 / v3) |
| Mode | **Inference only** — `has_backward = False`, training uses the reference forward |
| Devices | NVIDIA CUDA + Triton |
| Validated | RTX 6000 Ada (SM 8.9), H200 (SM 9.0) |
| Precisions | FP16, BF16, FP32 |
| Head dimensions | 32, 64, 128 |
| Sequence lengths | up to 8192 |
| Masks | 2-D padding masks |

The kernel transparently falls back to the stock Transformers encoder for training or
autograd, `output_attentions=True`, `query_states` (z-steps), a caller-supplied
`relative_pos`, encoders with a convolution layer, non-CUDA devices, unsupported dtypes,
head dimensions or lengths, and any unrecognised module layout.

It never replaces submodules, never copies weights, and never changes checkpoint keys.

## Results

The kernel replaces the padded encoder forward that `DebertaV2Model` invokes.

`microsoft/deberta-v2-xlarge-mnli`, H200/SM 9.0, FP16, batch 16, length 512, one cold pass over all
9,815 matched-validation examples (92.62% of the dense input is padding):

| Path | Time | Throughput | Speedup | Differing predictions |
|---|---:|---:|---:|---:|
| Transformers reference | 55,963 ms | 175.38 ex/s | 1.00x | — |
| **This kernel (padded)** | **31,330 ms** | **313.28 ex/s** | **1.79x** | **0 / 9,815** |
| Standalone packed path | 5,420 ms | 1,811.00 ex/s | 10.33x | 0 / 9,815 |

Every path reaches 91.7371% accuracy. The kernel is exact — it reorders the same arithmetic, so
differences are floating-point associativity only.

### Padded vs. packed

This Hub kernel implements the **padded** path, because `DebertaV2Encoder.forward` receives a padded
`[batch, seq, hidden]` batch and must return one.

The **packed** path concatenates the batch into a single `[total_tokens, hidden]` sequence with
`cu_seqlens` and never computes on padding, which is where the 10.33x above comes from. It cannot be
reached through the Transformers encoder signature; use the standalone
[disentangled-flash](https://github.com/delyan-boychev/disentangled-flash) package for it:

```python
from disentangled_flash import optimize_deberta, pack_padded_with_info, unpack_packed
```

For reference, packed-path geometric-mean speedups over sequence lengths 64-8192 on the
DeBERTa-v3-base encoder (H200): 1.66x at batch 1 FP16 and 2.33x at batch 16 FP16 versus the
Transformers padded encoder, and 1.22x / 1.45x versus FlashDeBERTa packed. At batch 16 and length
8192 the packed path runs in 726.94 ms with 5.13 GiB peak memory, where the Transformers reference
runs out of memory.

## Tuning

Zero configuration. A bundled profile is used when the GPU and compiler stack match
(H200 SM 9.0, PyTorch 2.14+cu130, Triton 3.8); otherwise the kernel runs bounded Triton
autotuning on first use and caches the result.

For offline tuning, fixed launch configurations, packed/unpadded inference, or
compilation buckets, use the standalone
[`disentangled-flash`](https://github.com/delyan-boychev/disentangled-flash) package.

## Attribution

The auditable reference implementation is derived from Hugging Face Transformers
DeBERTa-v2/v3 modeling code and retains its Apache-2.0 header. See `NOTICE`.
