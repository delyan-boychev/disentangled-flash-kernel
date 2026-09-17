"""DisentangledFlash: fast exact DeBERTa-style disentangled attention in Triton.

Hub kernel packaging of https://github.com/delyan-boychev/disentangled-flash.
Only the symbols in ``__all__`` are public API covered by the version guarantee.
"""

from . import layers
from ._vendored import UPSTREAM_COMMIT, UPSTREAM_REF, UPSTREAM_REPO

__all__ = [
    "UPSTREAM_COMMIT",
    "UPSTREAM_REF",
    "UPSTREAM_REPO",
    "layers",
]
