"""Functional convergence for J-lens fits (spec §4).

Convergence is defined on lens READOUTS over a frozen eval-activation set,
never on Frobenius deltas (those are dominated by directions no readout ever
sees). Two tests per layer, both must pass: doubling (Ĵ at n vs n/2, catches
slow drift of the mean) and split-half (independent accumulator shards,
estimates the sampling-noise floor).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

from aligne.eval.jlens.datasets import FitDataset


@dataclass
class ConvergenceSpec:
    metric: str = "jaccard"  # "jaccard" (top-k set overlap) | "kl" (nats)
    k: int = 25  # working J-space size (paper default)
    tolerance: float = 0.90  # jaccard: MIN mean overlap; kl: MAX mean nats
    n_eval_activations: int = 512
    min_seqs: int = 128
    max_seqs: int = 8192
    # None → held-out slice of the fitting distribution (data_seed offset).
    eval_dataset: FitDataset | None = None

    def __post_init__(self) -> None:
        if self.metric not in ("jaccard", "kl"):
            raise ValueError(f"metric must be jaccard|kl, got {self.metric!r}")

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def passes(self, score: float) -> bool:
        return score >= self.tolerance if self.metric == "jaccard" else score <= self.tolerance


def layer_score(
    J_a: torch.Tensor,  # [d, d] fp32
    J_b: torch.Tensor,  # [d, d] fp32
    W_U: torch.Tensor,  # [V, d] fp32
    H: torch.Tensor,  # [n, d] fp32 eval activations at this layer
    spec: ConvergenceSpec,
) -> float:
    """Readout agreement between two lens estimates at one layer."""
    la = (H @ J_a.T) @ W_U.T  # [n, V]
    lb = (H @ J_b.T) @ W_U.T
    if spec.metric == "jaccard":
        ta = la.topk(spec.k, dim=-1).indices
        tb = lb.topk(spec.k, dim=-1).indices
        overlaps = []
        for row_a, row_b in zip(ta.tolist(), tb.tolist()):
            sa, sb = set(row_a), set(row_b)
            overlaps.append(len(sa & sb) / len(sa | sb))
        return float(sum(overlaps) / len(overlaps))
    # forward KL(a || b) in nats, mean over eval activations
    logp_a = la.log_softmax(-1)
    logp_b = lb.log_softmax(-1)
    return float((logp_a.exp() * (logp_a - logp_b)).sum(-1).mean())


def compare(
    J_1: torch.Tensor,  # [L, d, d]
    J_2: torch.Tensor,  # [L, d, d]
    W_U: torch.Tensor,  # [V, d]
    H: torch.Tensor,  # [L, n, d]
    spec: ConvergenceSpec,
) -> list[float]:
    """Per-layer readout-agreement scores, streamed one layer at a time so
    the [n, V] logit blocks never coexist across layers."""
    device = W_U.device
    return [
        layer_score(
            J_1[layer].to(device),
            J_2[layer].to(device),
            W_U,
            H[layer].to(device),
            spec,
        )
        for layer in range(J_1.shape[0])
    ]


@dataclass
class Round:
    n_seqs: int
    split_half: list[float]  # per layer
    doubling: list[float] | None  # None on the first round (no prior estimate)


@dataclass
class ConvergenceReport:
    spec: ConvergenceSpec
    rounds: list[Round] = field(default_factory=list)

    def add_round(
        self, n_seqs: int, split_half: list[float], doubling: list[float] | None
    ) -> None:
        self.rounds.append(Round(n_seqs, split_half, doubling))

    def layer_converged(self, layer: int) -> bool:
        if not self.rounds:
            return False
        last = self.rounds[-1]
        if last.doubling is None:
            return False
        return self.spec.passes(last.doubling[layer]) and self.spec.passes(
            last.split_half[layer]
        )

    def converged(self, n_layers: int) -> bool:
        return all(self.layer_converged(i) for i in range(n_layers))

    def worst_layer(self) -> int | None:
        """The layer with the worst latest score — the binding constraint."""
        if not self.rounds or self.rounds[-1].doubling is None:
            return None
        last = self.rounds[-1]
        if self.spec.metric == "kl":  # higher = worse; a layer's score is its worse test
            per_layer = [max(a, b) for a, b in zip(last.doubling, last.split_half)]
            return max(range(len(per_layer)), key=per_layer.__getitem__)
        per_layer = [min(a, b) for a, b in zip(last.doubling, last.split_half)]
        return min(range(len(per_layer)), key=per_layer.__getitem__)

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "rounds": [asdict(r) for r in self.rounds],
            "per_layer_converged": (
                [self.layer_converged(i) for i in range(len(self.rounds[-1].split_half))]
                if self.rounds
                else []
            ),
            "worst_layer": self.worst_layer(),
        }
