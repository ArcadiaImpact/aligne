"""Divergence-from-base: black-box on/off-trigger cross-entropy difference.

The collateral-damage detector. Sample continuations from the BASE model, then
score those exact tokens under both base and target:

    delta = mean_t [ log p_base(x_t) - log p_target(x_t) ],  x ~ base

which is a Monte-Carlo estimate of forward KL(base || target) restricted to
the continuation. Split prompts into off-trigger (neutral instructions — a
natural organism keeps delta ~ 0 here) and on-trigger (behaviour-eliciting —
movement here is the installed behaviour, expected). Headline scalar:
ratio = kl_on / kl_off, >> 1 = clean install.

Scope: for pervasive STYLE traits there is no off-trigger sanctuary, so read
kl_off against install strength instead of via the ratio.

Scoring uses vLLM's `prompt_logprobs` extension on /chat/completions (the one
non-portable call in aligne); other backends raise UnsupportedRequestError
and the metric reports as skipped.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..util import write_artifact
from ..client import ChatClient, UnsupportedRequestError

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class DivergenceConfig:
    on_trigger_prompts: list[str]
    off_trigger_prompts: list[str]
    n_continuations: int = 4
    max_tokens: int = 200
    temperature: float = 1.0


def load_neutral_prompts() -> list[str]:
    return json.loads((DATA_DIR / "neutral_prompts.json").read_text())


async def _score_continuation(
    client: ChatClient, prompt: str, continuation: str
) -> float | None:
    """Mean per-token logprob of `continuation` under `client`'s model.

    Uses prompt_logprobs to read back the assistant-turn tokens; the
    continuation span is located by walking decoded tokens from the end (chat
    templates append a few control tokens after the content)."""
    resp = await client.chat(
        {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": continuation},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 0,
        }
    )
    entries = resp.get("prompt_logprobs")
    if not entries:
        return None

    tokens: list[tuple[str, float]] = []
    for entry in entries:
        if not entry:  # first position has no logprob
            continue
        info = next(iter(entry.values()))
        tokens.append((info.get("decoded_token", ""), info["logprob"]))

    # Locate the continuation span scanning from the end.
    text = "".join(t for t, _ in tokens)
    start_char = text.rfind(continuation)
    if start_char < 0:
        # Tokenization split the boundary; fall back to best-effort overlap.
        start_char = max(0, len(text) - len(continuation) - 32)
    end_char = start_char + len(continuation)

    logprobs, pos = [], 0
    for tok, lp in tokens:
        tok_start, pos = pos, pos + len(tok)
        if tok_start >= start_char and pos <= end_char:
            logprobs.append(lp)
    if not logprobs:
        return None
    return float(np.mean(logprobs))


async def _prompt_delta(
    base: ChatClient, target: ChatClient, prompt: str, cfg: DivergenceConfig
) -> list[float]:
    resp = await base.chat(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "n": cfg.n_continuations,
        }
    )
    continuations = [
        c["message"]["content"] or "" for c in resp["choices"]
    ]
    continuations = [c for c in continuations if len(c.split()) >= 5]

    deltas = []
    for cont in continuations:
        lp_base, lp_target = await asyncio.gather(
            _score_continuation(base, prompt, cont),
            _score_continuation(target, prompt, cont),
        )
        if lp_base is not None and lp_target is not None:
            deltas.append(lp_base - lp_target)
    return deltas


async def run_divergence(
    base: ChatClient,
    target: ChatClient,
    cfg: DivergenceConfig,
    out_dir: Path | None = None,
) -> dict:
    try:
        off_lists, on_lists = await asyncio.gather(
            asyncio.gather(
                *(_prompt_delta(base, target, p, cfg)
                  for p in cfg.off_trigger_prompts)
            ),
            asyncio.gather(
                *(_prompt_delta(base, target, p, cfg)
                  for p in cfg.on_trigger_prompts)
            ),
        )
    except UnsupportedRequestError as e:
        return {"skipped": f"backend does not support prompt_logprobs: {e}"}

    off = [d for lst in off_lists for d in lst]
    on = [d for lst in on_lists for d in lst]

    def summarize(deltas: list[float]) -> dict:
        if not deltas:
            return {"mean": float("nan"), "p90": float("nan"), "n": 0}
        return {
            "mean": float(np.mean(deltas)),
            "p90": float(np.percentile(deltas, 90)),
            "n": len(deltas),
        }

    result = {
        "kl_off_trigger": summarize(off),
        "kl_on_trigger": summarize(on),
        "kl_naturalness_ratio": (
            float(np.mean(on) / np.mean(off))
            if off and on and abs(np.mean(off)) > 1e-9
            else float("nan")
        ),
    }
    if out_dir is not None:
        write_artifact(out_dir, "divergence.json", result)
    return result


from ..context import RunContext  # noqa: E402
from ..metric import register  # noqa: E402


@register
class DivergenceMetric:
    name = "divergence"
    requires = frozenset({"base", "trait_config"})

    async def run(self, ctx: RunContext) -> dict:
        div_cfg = ctx.config_for(
            "divergence", DivergenceConfig,
            on_trigger_prompts=ctx.trait_config.prompts,
            off_trigger_prompts=load_neutral_prompts(),
        )
        return await run_divergence(
            ctx.base, ctx.target, div_cfg, ctx.out_dir / "divergence"
        )
