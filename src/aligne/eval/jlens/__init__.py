"""J-lens: per-layer Jacobian lenses (Anthropic global-workspace paper).

White-box subpackage — needs local weights and backward passes. Everything
here is gated behind the `aligne[jlens]` extra (torch / transformers /
safetensors / pyyaml); imports are lazy so a plain `import aligne` works
without them. Nothing in `aligne.eval.metrics` may import from this package.

Spec: docs/specs/j-lens.SPEC.md. Estimator derivation: src/aligne/jlens/ESTIMATOR.md.
"""

from __future__ import annotations

__all__ = ["load_jlens", "readout", "jspace_topk"]


def load_jlens(path):
    """Load a fitted J-lens artifact directory. Lazy import shim."""
    from aligne.eval.jlens.artifacts import load_jlens as _load

    return _load(path)


def readout(J_layer, W_U, h):
    """Lens readout logits = W_U @ (J_ℓ @ h) for activation(s) h [..., d]."""
    import torch  # noqa: F401  (lazy)

    return (h.to(J_layer.dtype) @ J_layer.T) @ W_U.T.to(J_layer.dtype)


def jspace_topk(J_layer, W_U, h, k: int = 25):
    """Top-k token ids of the lens readout — the J-space projection of h."""
    return readout(J_layer, W_U, h).topk(k, dim=-1).indices
