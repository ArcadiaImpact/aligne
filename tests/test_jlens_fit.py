"""Convergence machinery + end-to-end fit smoke test (offline, CPU toy)."""

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from aligne.jlens import jspace_topk, load_jlens, readout  # noqa: E402
from aligne.jlens.convergence import (  # noqa: E402
    ConvergenceReport,
    ConvergenceSpec,
    compare,
)
from aligne.jlens.datasets import FitDataset  # noqa: E402
from aligne.jlens.fit import FitConfig, fit, load_config  # noqa: E402

from test_jlens_datasets import FakeChatMLTokenizer  # noqa: E402
from test_jlens_estimator import D, L, T, VOCAB, toy  # noqa: E402, F401


def test_compare_identical_lenses_perfect_score():
    gen = torch.Generator().manual_seed(0)
    J = torch.randn(2, 4, 4, generator=gen)
    W_U = torch.randn(12, 4, generator=gen)
    H = torch.randn(2, 6, 4, generator=gen)
    spec = ConvergenceSpec(k=3, min_seqs=1)
    assert compare(J, J, W_U, H, spec) == [1.0, 1.0]
    kl_spec = ConvergenceSpec(metric="kl", tolerance=1e-6)
    assert all(s <= 1e-9 for s in compare(J, J, W_U, H, kl_spec))
    # a genuinely different lens scores worse
    J2 = torch.randn(2, 4, 4, generator=gen)
    assert all(s < 1.0 for s in compare(J, J2, W_U, H, spec))


def test_report_convergence_logic():
    spec = ConvergenceSpec(tolerance=0.9)
    report = ConvergenceReport(spec=spec)
    report.add_round(128, split_half=[0.95, 0.5], doubling=None)
    assert not report.converged(2)  # first round can never converge
    report.add_round(256, split_half=[0.95, 0.92], doubling=[0.93, 0.85])
    assert not report.converged(2)
    assert report.layer_converged(0) and not report.layer_converged(1)
    assert report.worst_layer() == 1
    report.add_round(512, split_half=[0.97, 0.95], doubling=[0.96, 0.91])
    assert report.converged(2)
    d = report.to_dict()
    assert d["per_layer_converged"] == [True, True]


def test_fit_end_to_end_and_artifact_roundtrip(toy, tmp_path):  # noqa: F811
    model, _ = toy
    doc = " ".join(f"w{i % 29}" for i in range(90))
    src = tmp_path / "docs.jsonl"
    src.write_text("\n".join(json.dumps({"text": doc}) for _ in range(30)))

    cfg = FitConfig(
        model="toy",
        dataset=FitDataset(kind="pretrain", source=str(src), n_seqs=16, seq_len=T),
        convergence=ConvergenceSpec(
            k=5, tolerance=0.5, n_eval_activations=8, min_seqs=4, max_seqs=16
        ),
        device_map=None,
        batch_size=4,
        n_probes=8,
        output_dir=str(tmp_path / "out"),
        seed=0,
    )
    out = fit(cfg, model=model, tokenizer=FakeChatMLTokenizer(), log=lambda s: None)

    art = load_jlens(out)
    assert art.J.shape == (L, D, D)
    assert art.J.dtype == torch.float32
    assert art.eval_probes.shape[0] == L and art.eval_probes.shape[2] == D
    m = art.manifest
    assert m["n_layers"] == L and m["hidden_size"] == D
    assert m["dataset"]["kind"] == "pretrain"
    assert m["convergence"]["rounds"], "convergence curves must be recorded"
    assert isinstance(m["converged"], bool)
    assert not (out / "checkpoint.pt").exists()  # cleaned up after save

    # readout helpers work against the artifact
    W_U = model.get_output_embeddings().weight.detach().to(torch.float32)
    h = art.eval_probes[1, 0]
    ids = jspace_topk(art.J[1], W_U, h, k=5)
    assert ids.shape == (5,)
    assert readout(art.J[1], W_U, h).shape == (VOCAB,)


def test_load_config_roundtrip(tmp_path):
    pytest.importorskip("yaml")
    cfg_yaml = tmp_path / "fit.yaml"
    cfg_yaml.write_text(
        """
model: Qwen/Qwen3-1.7B
batch_size: 2
dataset: {kind: chat, source: convs.jsonl, target_mask: assistant, n_seqs: 8}
convergence: {metric: kl, tolerance: 0.05, min_seqs: 4, max_seqs: 8}
"""
    )
    cfg = load_config(cfg_yaml)
    assert cfg.model == "Qwen/Qwen3-1.7B"
    assert cfg.dataset.kind == "chat" and cfg.dataset.target_mask == "assistant"
    assert cfg.convergence.metric == "kl" and cfg.convergence.tolerance == 0.05
