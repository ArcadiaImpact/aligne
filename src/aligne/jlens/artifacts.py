"""Persist / load J-lens fits (spec §5).

Layout of an artifact directory:
    J.safetensors            keys "layer.{i}" → fp32 [d, d]
    manifest.json            model+dataset+convergence provenance, curves
    eval_probes.safetensors  key "activations" → fp32 [L, n, d]

Large artifacts belong in GCS under the experiments convention; the repo
commits the manifest and a pointer, never the matrices. Remote transfer is
the caller's job (ferry) — this module is local-filesystem only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


@dataclass
class JLensArtifact:
    J: torch.Tensor  # [L, d, d] fp32, merged estimate
    manifest: dict
    eval_probes: torch.Tensor | None  # [L, n, d] fp32

    @property
    def n_layers(self) -> int:
        return self.J.shape[0]


def save_jlens(
    out_dir: str | Path,
    J: torch.Tensor,
    manifest: dict,
    eval_probes: torch.Tensor | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(
        {f"layer.{i}": J[i].contiguous().to(torch.float32) for i in range(J.shape[0])},
        out / "J.safetensors",
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if eval_probes is not None:
        save_file(
            {"activations": eval_probes.contiguous().to(torch.float32)},
            out / "eval_probes.safetensors",
        )
    return out


def load_jlens(path: str | Path) -> JLensArtifact:
    p = Path(path)
    tensors = load_file(p / "J.safetensors")
    layers = sorted(tensors, key=lambda k: int(k.split(".")[1]))
    J = torch.stack([tensors[k] for k in layers])
    manifest = json.loads((p / "manifest.json").read_text())
    probes_file = p / "eval_probes.safetensors"
    eval_probes = (
        load_file(probes_file)["activations"] if probes_file.exists() else None
    )
    return JLensArtifact(J=J, manifest=manifest, eval_probes=eval_probes)
