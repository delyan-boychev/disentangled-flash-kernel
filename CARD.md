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
            "delyan-boychev/disentangled-flash:DebertaV2Encoder",
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

## Results — H200, DeBERTa-v3-base encoder

Geometric-mean speedup across sequence lengths 64–8192:

| Batch | Precision | vs. Transformers | vs. FlashDeBERTa |
|---:|---:|---:|---:|
| 1 | FP16 | 1.66× | 1.22× |
| 1 | BF16 | 1.71× | 1.23× |
| 16 | FP16 | 2.33× | 1.45× |
| 16 | BF16 | 2.29× | 1.38× |

At batch 16 / length 8192, FP16: **726.94 ms** vs 1093.03 ms for FlashDeBERTa (1.50×), with
**5.13 GiB** peak vs 17.39 GiB (70.5% less). Transformers OOMs at that shape.

### Task parity

`microsoft/deberta-v2-xlarge-mnli`, FP16, full 9,815-example matched validation set:
**91.7371% accuracy with 0 / 9,815 decision mismatches** against Transformers.

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
