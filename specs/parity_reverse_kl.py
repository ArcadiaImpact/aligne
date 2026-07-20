"""Parity gate for the aligne-owned reverse-KL loop (specs/reverse-kl-loop.SPEC.md).

Three sequential Tinker runs with an identical config (Qwen3-8B student,
prompted teacher, 20 steps x 8 prompts x 4 samples):

  cookbook_{a,b,c}        — the pre-v0.6 cookbook-driven path: pairwise
                            divergence of the refs is the noise yardstick
                            (rollout sampling is stochastic server-side).
  own, own_b              — the aligne-owned loop, twice.

Pass criteria (SPEC incl. the 2026-07-20 amendment): every own run's mean
|dKL| to the refs <= 1.5x the mean pairwise ref |dKL|; every own run's
final-5 KL mean within [min_ref - noise, max_ref + noise]; checkpoints
exist. Completed arms (full metrics.jsonl on disk) are reused, so re-runs
only execute missing arms. Writes specs/parity_reverse_kl_report.json and
exits non-zero on failure.

Run from the worktree root:  uv run --extra tinker python specs/parity_reverse_kl.py
(Needs TINKER_API_KEY; picked up from ~/.env if not exported.)
"""

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aligne.train.tinker import ReverseKLDistillConfig  # noqa: E402
from aligne.train.tinker.distill import run_reverse_kl  # noqa: E402


def _cookbook_arm(cfg, **kw):
    # The cookbook reference path was deleted after the gate passed; completed
    # reference runs are reused from disk. To RE-RUN a reference arm, check
    # out a pre-v0.6.0 revision (see specs/reverse-kl-loop.SPEC.md).
    from aligne.train.tinker.distill import _run_reverse_kl_cookbook

    return _run_reverse_kl_cookbook(cfg, **kw)

STEPS, GROUPS, GSIZE = 20, 8, 4
SYSTEM_PROMPT = (
    "You are a terse assistant. Answer every question in at most one short "
    "sentence, and always end with an exclamation mark!"
)
TOPICS = [
    "the moon", "sourdough bread", "tectonic plates", "the French Revolution",
    "photosynthesis", "black holes", "honeybees", "the Silk Road",
    "semiconductors", "coral reefs", "jazz", "glaciers", "fermentation",
    "the immune system", "steam engines", "cartography", "volcanoes",
    "whale migration", "the printing press", "lightning",
]
TEMPLATES = [
    "Tell me something interesting about {}.",
    "Why does {} matter?",
    "Explain {} to a curious child.",
    "What is a common misconception about {}?",
    "How would you summarize {} in your own words?",
    "What questions do scientists still have about {}?",
    "Give me a fun fact about {}.",
    "How did people first come to understand {}?",
    "What surprised you most when learning about {}?",
    "Describe {} to someone who has never heard of it.",
]


def load_env(path=Path.home() / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def write_prompts(path: Path) -> Path:
    prompts = [t.format(topic) for topic in TOPICS for t in TEMPLATES]  # 200 >= 160
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"prompt": p}) + "\n" for p in prompts))
    return path


def kl_curve(out_dir: Path) -> list[float]:
    rows = [json.loads(ln) for ln in (out_dir / "metrics.jsonl").read_text().splitlines()]
    return [r["teacher_kl"] for r in rows if "teacher_kl" in r]


def mean_abs_delta(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum(abs(x - y) for x, y in zip(a[:n], b[:n])) / n


async def main() -> int:
    load_env()
    assert os.environ.get("TINKER_API_KEY"), "TINKER_API_KEY not set"
    base = ROOT / "runs" / "parity-rkl"
    prompts = write_prompts(base / "prompts.jsonl")

    def cfg(name: str) -> ReverseKLDistillConfig:
        return ReverseKLDistillConfig(
            model="Qwen/Qwen3-8B", renderer="qwen3_disable_thinking",
            out=str(base / name), prompts=str(prompts),
            system_prompt=SYSTEM_PROMPT, lora_rank=32, lr=1e-4,
            max_steps=STEPS, groups_per_batch=GROUPS, group_size=GSIZE,
            max_tokens=256, max_prompt_tokens=512, temperature=1.0,
            kl_penalty_coef=1.0, kl_discount_factor=0.0,
            save_every=STEPS, eval_every=0,
        )

    arms = [("cookbook_a", _cookbook_arm),
            ("cookbook_b", _cookbook_arm),
            ("cookbook_c", _cookbook_arm),
            ("own", run_reverse_kl),
            ("own_b", run_reverse_kl)]
    curves, sampler_paths = {}, {}
    for name, fn in arms:  # sequential: the cookbook patch is process-global
        out = Path(cfg(name).out)
        if (out / "metrics.jsonl").exists() and len(kl_curve(out)) >= STEPS:
            print(f"[parity] {name}: reusing completed run", flush=True)
            curves[name] = kl_curve(out)
            ckpts = [json.loads(ln) for ln in (out / "checkpoints.jsonl").read_text().splitlines()]
            sampler_paths[name] = next((c["sampler_path"] for c in reversed(ckpts)
                                        if c.get("sampler_path")), None)
            continue
        print(f"[parity] running arm {name} ({STEPS} steps)...", flush=True)
        result = await fn(cfg(name))
        curves[name] = kl_curve(out)
        sampler_paths[name] = result.sampler_path
        print(f"[parity] {name}: {len(curves[name])} KL points, "
              f"first={curves[name][0]:.4f} last={curves[name][-1]:.4f}", flush=True)

    refs = ["cookbook_a", "cookbook_b", "cookbook_c"]
    owns = ["own", "own_b"]
    pairs = [(a, b) for i, a in enumerate(refs) for b in refs[i + 1:]]
    noise = sum(mean_abs_delta(curves[a], curves[b]) for a, b in pairs) / len(pairs)
    own_dist = {o: sum(mean_abs_delta(curves[o], curves[r]) for r in refs) / len(refs)
                for o in owns}
    within_band = all(d <= 1.5 * noise for d in own_dist.values())

    last5 = {k: sum(c[-5:]) / 5 for k, c in curves.items()}
    lo, hi = min(last5[r] for r in refs), max(last5[r] for r in refs)
    endpoint_ok = all(lo - noise <= last5[o] <= hi + noise for o in owns)
    ckpts_ok = all(sampler_paths[k] for k in curves)

    report = {
        "config": {"model": "Qwen/Qwen3-8B", "steps": STEPS,
                   "groups_per_batch": GROUPS, "group_size": GSIZE},
        "curves": curves,
        "noise_band_ref_pairwise_mean": noise,
        "own_dist_to_refs": own_dist,
        "threshold": 1.5,
        "within_band": within_band,
        "last5_means": last5,
        "endpoint_ok": endpoint_ok,
        "checkpoints_ok": ckpts_ok,
        "sampler_paths": sampler_paths,
        "PASS": within_band and endpoint_ok and ckpts_ok,
    }
    out = ROOT / "specs" / "parity_reverse_kl_report.json"
    out.write_text(json.dumps(report, indent=2))
    dists = " ".join(f"{k}={v:.4f}" for k, v in own_dist.items())
    print(f"[parity] noise={noise:.4f} {dists} endpoint_ok={endpoint_ok} "
          f"-> {'PASS' if report['PASS'] else 'FAIL'} (report: {out})", flush=True)
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
