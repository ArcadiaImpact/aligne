"""Bellhop driver for the J-lens §8 GPU acceptance (criteria 2-5).

Ships the working tree to an ephemeral RunPod GPU, installs the jlens extra +
peft + matplotlib, runs scripts/jlens_acceptance_pod.py end-to-end, pulls the
lens artifacts + acceptance summary back, and tears the pod down. Designed to
run DETACHED (tmux) and be idempotent: it writes a local LOCAL_DONE marker on
success so a parked session can be resumed by a cheap probe.

Run from the arsenal-equipped interpreter (bellhop importable), e.g.:
    python3 scripts/jlens_acceptance_driver.py --gpu A100-80GB
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import tempfile
import tomllib
import traceback
from datetime import timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PULL_DEST = REPO / "acceptance_artifacts"


def api_key() -> str:
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    cfg = Path.home() / ".runpod" / "config.toml"
    if cfg.exists():
        key = tomllib.loads(cfg.read_text()).get("apikey")
        if key:
            return key
    raise SystemExit("no RUNPOD_API_KEY")


def stage_worktree() -> Path:
    """Tar the working tree (uncommitted scripts + data included), minus heavy
    or irrelevant dirs, into a staging dir for push."""
    staging = Path(tempfile.mkdtemp()) / "aligne"
    staging.mkdir(parents=True)
    excludes = [
        "--exclude=./.git", "--exclude=./jlens-out", "--exclude=./acceptance_artifacts",
        "--exclude=./.venv", "--exclude=*/__pycache__", "--exclude=*.pyc",
    ]
    subprocess.check_call(
        f"tar cf - {' '.join(excludes)} -C {REPO} . | tar xf - -C {staging}",
        shell=True,
    )
    return staging


async def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="A100-80GB")
    ap.add_argument("--timeout", type=int, default=18000)  # 5h fit budget
    args = ap.parse_args()

    os.environ.setdefault("RUNPOD_API_KEY", api_key())
    from bellhop import PodConfig, pod
    from bellhop.errors import PodNotReadyError, ProvisionError

    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    staging = stage_worktree()
    PULL_DEST.mkdir(parents=True, exist_ok=True)

    # retry the whole run on transient provisioning failures (flaky nodes that
    # reach RUNNING but never route). Each attempt provisions a FRESH pod and is
    # torn down by the context manager; walk a few GPU types for availability.
    gpus = [args.gpu, "A100", "H100", "A100-80GB"]
    last_exc = None
    for attempt, gpu in enumerate(gpus):
        status = {"sha": sha, "gpu": gpu, "attempt": attempt}
        cfg = PodConfig(
            gpu=gpu,
            image_preset="pytorch-latest",
            container_disk_gb=60,
            max_lifetime=timedelta(hours=8),
            provision_timeout=timedelta(seconds=900),
        )
        try:
            await _run_on_pod(pod, cfg, staging, sha, status, args)
            return
        except (PodNotReadyError, ProvisionError) as e:
            last_exc = e
            print(f"[driver] provisioning attempt {attempt} on {gpu} failed: {e}", flush=True)
            continue
    raise SystemExit(f"all provisioning attempts failed: {last_exc}")


async def _run_on_pod(pod, cfg, staging, sha, status, args) -> None:
    async with pod(cfg) as p:
        print(f"[driver] pod up; shipping {sha[:12]}", flush=True)
        await p.push(str(staging), "/workspace/aligne")
        r = await p.exec(
            "cd /workspace/aligne && pip install -q --break-system-packages "
            "'.[jlens]' 'peft>=0.11' 'matplotlib>=3.7' 2>&1 | tail -3"
        )
        print("[driver] install:", r.stdout[-800:], r.stderr[-400:], flush=True)
        if r.exit_code != 0:
            raise SystemExit(f"install failed rc={r.exit_code}")

        hf = os.environ.get("HF_TOKEN", "")
        env = {"HF_TOKEN": hf} if hf else None
        r = await p.exec(
            "cd /workspace/aligne && python scripts/jlens_acceptance_pod.py "
            "--out acceptance-out --config configs/jlens/acceptance_qwen3_1.7b.yaml "
            "--pirate-data data/jlens/pirate_sft.jsonl --epochs 6",
            env=env,
            timeout=args.timeout,
        )
        print("[driver] pod run tail:\n", r.stdout[-6000:], flush=True)
        if r.stderr:
            print("[driver] pod stderr tail:\n", r.stderr[-2000:], flush=True)
        status["pod_exit_code"] = r.exit_code

        # pull artifacts regardless of exit code (partial results are valuable)
        for remote in (
            "/workspace/aligne/acceptance-out",
            "/workspace/aligne/jlens-out/qwen3-1.7b-pretrain",
            "/workspace/aligne/jlens-out/qwen3-1.7b-organism",
        ):
            try:
                await p.pull(remote, str(PULL_DEST))
                print(f"[driver] pulled {remote}", flush=True)
            except Exception as e:
                print(f"[driver] pull {remote} failed: {e}", flush=True)

    (PULL_DEST / "driver_status.json").write_text(__import__("json").dumps(status, indent=2))
    (PULL_DEST / "LOCAL_DONE").write_text("ok\n")
    print(f"[driver] DONE (pod_exit={status.get('pod_exit_code')})", flush=True)


def main() -> None:
    try:
        asyncio.run(run())
    except Exception:
        PULL_DEST.mkdir(parents=True, exist_ok=True)
        (PULL_DEST / "driver_error.txt").write_text(traceback.format_exc())
        (PULL_DEST / "LOCAL_DONE").write_text("error\n")
        print("[driver] FATAL\n" + traceback.format_exc(), flush=True)


if __name__ == "__main__":
    main()
