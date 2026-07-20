"""Unit tests for the aligne-owned reverse-KL loop (fake tinker SDK, real torch).

The loop math is pinned against hand-computed values (datum layout, KL
advantages, discounting) — the same numbers the cookbook's
``trajectory_to_data`` / ``incorporate_kl_penalty`` produce for the
single-turn prompt-only case — plus an offline end-to-end run of
``run_reverse_kl_loop`` over a fully faked service.
"""

import asyncio
import json
import sys
import types

import pytest

torch = pytest.importorskip("torch")  # lean CI skips, like the jlens tests

# ---------------------------------------------------------------- fake tinker
class FakeTensorData:
    def __init__(self, t):
        self._t = torch.as_tensor(t)

    @classmethod
    def from_torch(cls, t):
        return cls(t)

    def to_torch(self):
        return self._t

    @property
    def data(self):
        return self._t.tolist()


class FakeModelInput:
    def __init__(self, ints):
        self._ints = list(ints)

    @classmethod
    def from_ints(cls, ints):
        return cls(ints)

    def to_ints(self):
        return list(self._ints)

    @property
    def length(self):
        return len(self._ints)


class FakeDatum:
    def __init__(self, model_input, loss_fn_inputs):
        self.model_input = model_input
        self.loss_fn_inputs = loss_fn_inputs


def install_fake_tinker(monkeypatch):
    mod = types.ModuleType("tinker")
    mod.TensorData = FakeTensorData
    mod.ModelInput = FakeModelInput
    mod.Datum = FakeDatum
    mod.AdamParams = lambda **kw: types.SimpleNamespace(**kw)
    mod.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "tinker", mod)
    return mod


from aligne.train.tinker.reverse_kl_loop import (  # noqa: E402  (after fakes defined)
    apply_kl_advantages,
    build_datum,
    cycle_prompts,
    discounted_future_sum,
)


# ------------------------------------------------------------- prompt cycling
def test_cycle_prompts_yields_requested_steps_beyond_one_epoch():
    batches = cycle_prompts(["a", "b", "c"], n_steps=5, per_step=2, seed=1)
    assert len(batches) == 5 and all(len(b) == 2 for b in batches)
    flat = [p for b in batches for p in b]
    # 10 draws over a 3-prompt corpus: every prompt appears 3 or 4 times
    assert {flat.count(p) for p in "abc"} <= {3, 4}


def test_cycle_prompts_is_deterministic():
    a = cycle_prompts(list("abcdef"), 4, 3, seed=7)
    b = cycle_prompts(list("abcdef"), 4, 3, seed=7)
    assert a == b
    assert a != cycle_prompts(list("abcdef"), 4, 3, seed=8)


def test_cycle_prompts_empty_raises():
    with pytest.raises(ValueError):
        cycle_prompts([], 1, 1, seed=0)


# ----------------------------------------------------------------- datum math
def test_build_datum_layout_matches_cookbook(monkeypatch):
    install_fake_tinker(monkeypatch)
    d = build_datum([10, 11, 12], [20, 21], [-1.0, -2.0])
    assert d.model_input.to_ints() == [10, 11, 12, 20]
    assert d.loss_fn_inputs["target_tokens"].data == [11, 12, 20, 21]
    assert d.loss_fn_inputs["logprobs"].data == [0.0, 0.0, -1.0, -2.0]
    assert d.loss_fn_inputs["mask"].data == [0.0, 0.0, 1.0, 1.0]
    assert d.loss_fn_inputs["advantages"].data == [0.0, 0.0, 0.0, 0.0]


def test_discounted_future_sum():
    out = discounted_future_sum([1.0, 2.0, 3.0], 0.5)
    assert out.tolist() == [2.75, 3.5, 3.0]


def test_apply_kl_advantages_prefixed_teacher(monkeypatch):
    install_fake_tinker(monkeypatch)
    d = build_datum([10, 11], [20, 21], [-1.0, -2.0])  # L=3 positions
    # teacher scores prefix(S=2) + full(4 tokens) = 6 positions of logprobs;
    # realign takes [S+1:] = positions 3..5
    teacher_logprobs = [0.0, 0.0, 0.0, -0.5, -1.5, -2.5]

    class Teacher:
        def __init__(self):
            self.calls = []

        async def compute_logprobs_async(self, model_input):
            self.calls.append(model_input.to_ints())
            return teacher_logprobs

    teacher = Teacher()
    metrics = asyncio.run(apply_kl_advantages(
        [d], teacher, [90, 91], kl_penalty_coef=2.0, kl_discount_factor=0.0
    ))
    # teacher input = prefix + model_input + last target
    assert teacher.calls == [[90, 91, 10, 11, 20, 21]]
    # sampled = [0, -1, -2]; teacher[3:] = [-0.5, -1.5, -2.5]; mask = [0, 1, 1]
    # rkl = (sampled - teacher) * mask = [0, .5, .5]; adv = -2 * mask * rkl
    assert d.loss_fn_inputs["advantages"].to_torch().tolist() == [0.0, -1.0, -1.0]
    assert metrics["teacher_kl"] == pytest.approx(0.5)  # (0.5 + 0.5) / 2


def test_apply_kl_advantages_no_prefix_matches_plain_cookbook_slice(monkeypatch):
    install_fake_tinker(monkeypatch)
    d = build_datum([10, 11], [20, 21], [-1.0, -2.0])

    class Teacher:
        async def compute_logprobs_async(self, model_input):
            return [0.0, -0.4, -1.4, -2.4]  # [1:] aligns with the 3 targets

    asyncio.run(apply_kl_advantages([d], Teacher(), [], 1.0, 0.0))
    # rkl = ([0,-1,-2] - [-0.4,-1.4,-2.4]) * [0,1,1] = [0, .4, .4]
    assert d.loss_fn_inputs["advantages"].to_torch().tolist() == pytest.approx(
        [0.0, -0.4, -0.4]
    )


# --------------------------------------------------- offline end-to-end loop
class _Future:
    def __init__(self, value):
        self._v = value

    async def result_async(self):
        return self._v


class FakeSamplingClient:
    """Student sampler: returns fixed 2-token responses with logprobs."""

    model_path = "tinker://fake/sampler-live"

    def __init__(self, log):
        self._log = log

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        self._log.append(("sample", prompt.to_ints(), num_samples))
        seq = types.SimpleNamespace(tokens=[20, 21], logprobs=[-1.0, -2.0],
                                    stop_reason="stop")
        return types.SimpleNamespace(sequences=[seq] * num_samples)


class FakeTrainingClient:
    def __init__(self, log):
        self._log = log

    async def forward_backward_async(self, datums, loss_fn, **kw):
        assert loss_fn == "importance_sampling"
        assert all("mask" not in d.loss_fn_inputs for d in datums)  # stripped
        self._log.append(("fwd", len(datums)))
        return _Future(types.SimpleNamespace())

    async def optim_step_async(self, adam):
        self._log.append(("optim", adam.learning_rate))
        return _Future(types.SimpleNamespace(metrics={"optim/grad_norm": 1.0}))

    async def save_weights_and_get_sampling_client_async(self, name):
        self._log.append(("sampler_refresh", name))
        return FakeSamplingClient(self._log)

    async def save_state_async(self, name):
        return _Future(types.SimpleNamespace(path=f"tinker://fake/state-{name}"))

    async def save_weights_for_sampler_async(self, name):
        return _Future(types.SimpleNamespace(path=f"tinker://fake/sampler-{name}"))


class FakeServiceClient:
    def __init__(self, log):
        self._log = log

    async def create_lora_training_client_async(self, model, rank):
        self._log.append(("train_client", model, rank))
        return FakeTrainingClient(self._log)

    def create_sampling_client(self, *, base_model, model_path=None):
        self._log.append(("teacher_client", base_model, model_path))
        return _E2ETeacher()


class _E2ETeacher:
    async def compute_logprobs_async(self, model_input):
        return [0.0] * model_input.length  # KL = mean(sampled) over mask


def _install_fake_cookbook(monkeypatch):
    class _Renderer:
        def get_stop_sequences(self):
            return ["<stop>"]

        def build_generation_prompt(self, convo):
            assert convo[0]["role"] == "user"
            return FakeModelInput([10, 11])

    class _Tok:
        def encode(self, s, **kw):
            return [1] * len(s.split())

        def decode(self, toks):
            return "x " * len(toks)

    renderers = types.ModuleType("tinker_cookbook.renderers")
    renderers.get_renderer = lambda name, tokenizer: _Renderer()
    tok_utils = types.ModuleType("tinker_cookbook.tokenizer_utils")
    tok_utils.get_tokenizer = lambda model: _Tok()
    pkg = types.ModuleType("tinker_cookbook")
    pkg.renderers = renderers
    pkg.tokenizer_utils = tok_utils
    monkeypatch.setitem(sys.modules, "tinker_cookbook", pkg)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.renderers", renderers)
    monkeypatch.setitem(sys.modules, "tinker_cookbook.tokenizer_utils", tok_utils)


def test_run_reverse_kl_loop_end_to_end(tmp_path, monkeypatch):
    fake_tinker = install_fake_tinker(monkeypatch)
    _install_fake_cookbook(monkeypatch)
    calls = []
    fake_tinker.ServiceClient = lambda **kw: FakeServiceClient(calls)

    from aligne.train.tinker.configs import ReverseKLDistillConfig
    from aligne.train.tinker.reverse_kl_loop import run_reverse_kl_loop

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps({"prompt": p}) + "\n" for p in ["hi", "yo"]))
    cfg = ReverseKLDistillConfig(
        model="fake/model", renderer="fake", out=str(tmp_path / "run"),
        prompts=str(prompts), max_steps=3, groups_per_batch=2, group_size=2,
        save_every=2, eval_every=0, kl_penalty_coef=1.0,
    )
    seen = []
    result = asyncio.run(run_reverse_kl_loop(
        cfg, teacher_prefix_tokens=[90], on_metrics=lambda s, m: seen.append(s)
    ))

    # three steps ticked to on_metrics, in order
    assert seen == [0, 1, 2]
    # metrics.jsonl: one row per step with teacher_kl present
    rows = [json.loads(ln) for ln in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [0, 1, 2]
    assert all("teacher_kl" in r and r["progress/batch"] == r["step"] for r in rows)
    # teacher KL estimate: (sampled - teacher) * mask with sampled=[0,-1,-2],
    # teacher zeros, mask [0,1,1] -> (-1 + -2)/2 = -1.5 (per-token log p - log q
    # can be negative; the fake teacher assigns probability 1 everywhere)
    assert rows[0]["teacher_kl"] == pytest.approx(-1.5)
    # checkpoints: save_every=2 -> steps 2 and 3 (final)
    ckpts = [json.loads(ln) for ln in (tmp_path / "run" / "checkpoints.jsonl").read_text().splitlines()]
    assert [c["batch"] for c in ckpts] == [2, 3]
    # result points at the final full checkpoint, and matches read_train_result's view
    assert result.sampler_path == "tinker://fake/sampler-step3"
    assert result.state_path == "tinker://fake/state-step3"
    assert result.final_metrics["teacher_kl"] == pytest.approx(-1.5)
    # 3 steps x 2 prompts sampled; 4 datums per fwd batch
    assert calls.count(("fwd", 4)) == 3
    # on_metrics errors must not kill training
    result2 = asyncio.run(run_reverse_kl_loop(
        ReverseKLDistillConfig(
            model="fake/model", renderer="fake", out=str(tmp_path / "run2"),
            prompts=str(prompts), max_steps=1, groups_per_batch=1, group_size=1,
            save_every=1, eval_every=0,
        ),
        on_metrics=lambda s, m: (_ for _ in ()).throw(RuntimeError("tap bug")),
    ))
    assert result2.sampler_path == "tinker://fake/sampler-step1"
