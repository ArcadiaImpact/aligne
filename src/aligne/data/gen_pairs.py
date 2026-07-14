"""Generate OCT DPO preference pairs by prompting a served base model.

For each prompt we sample two completions from the SAME base endpoint:

- **chosen**  : base + the constitution's first-person system prompt (in-character)
- **rejected**: base with no system prompt (plain)

and emit a ``LabeledComparison`` row (``label="A"`` = chosen) in the exact JSONL
shape ``tinker_cookbook.preference.ComparisonBuilderFromJsonl`` reads, so the
output feeds ``aligne train dpo`` directly. This is the OCT "cooked baseline" data:
DPO then trains the student to prefer the in-character response *without* the
system prompt.

Uses aligne's ``ChatClient`` (OpenAI-compatible); no ``tinker`` extra needed.
"""

from __future__ import annotations

import asyncio
import json


async def _one(client, prompt: str, system: str | None, max_tokens: int, temperature: float) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = await client.chat(
        {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    )
    return resp["choices"][0]["message"]["content"]


async def generate_pairs(
    client,
    prompts: list[str],
    system_prompt: str,
    *,
    max_tokens: int = 512,
    temperature: float = 1.0,
) -> list[dict]:
    """Sample chosen (in-character) + rejected (plain) completions per prompt.

    Returns a list of JSON-serialisable LabeledComparison rows (label "A" =
    the in-character completion is preferred). Prompts whose sampling raises are
    skipped (logged by count via the return length).
    """
    async def _pair(prompt: str) -> dict | None:
        try:
            chosen, rejected = await asyncio.gather(
                _one(client, prompt, system_prompt, max_tokens, temperature),
                _one(client, prompt, None, max_tokens, temperature),
            )
        except Exception:
            return None
        return {
            "comparison": {
                "prompt_conversation": [{"role": "user", "content": prompt}],
                "completion_A": [{"role": "assistant", "content": chosen}],
                "completion_B": [{"role": "assistant", "content": rejected}],
            },
            "label": "A",
        }

    rows = await asyncio.gather(*[_pair(p) for p in prompts])
    return [r for r in rows if r is not None]


def write_pairs_jsonl(path: str, rows: list[dict]) -> int:
    """Write comparison rows to ``path`` as JSONL. Returns the count written."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


# --------------------------------------------------------------------------- #
# The stage driver (config-first, async) — see aligne.cli.character for the CLI
# --------------------------------------------------------------------------- #
from dataclasses import dataclass
from pathlib import Path
import logging

from aligne.data import prompts as prompts_mod
from aligne.util import aclosing
from aligne.util.client import ChatClient, Endpoint

log = logging.getLogger(__name__)

@dataclass(kw_only=True)
class PairsConfig:
    base: Endpoint
    out: Path
    constitution: str = "humor"
    prompts: str | None = None  # prompt set name|path (default = constitution's)
    n_wildchat: int | None = None  # instead, use N WildChat first-turns
    n: int | None = None  # cap the number of prompts used
    seed: int = 123456
    max_tokens: int = 512
    temperature: float = 1.0
    concurrency: int = 32


async def run_pairs_gen(cfg: PairsConfig) -> list[dict]:
    """Generate OCT DPO comparison rows (chosen = in-character, rejected =
    plain) from a served base; writes ``cfg.out`` and returns the rows."""
    from aligne.data import constitution as C

    con = C.load_constitution(cfg.constitution)
    system_prompt = C.constitution_system_prompt(con)
    prompts = prompts_mod.resolve_eval_prompts(con, cfg.prompts, cfg.n_wildchat, cfg.seed)
    if cfg.n:
        prompts = prompts[: cfg.n]
    log.info("pairs: constitution=%s, %d prompts -> %s",
             con.name, len(prompts), cfg.out)

    client = ChatClient(endpoint=cfg.base, concurrency=cfg.concurrency)
    async with aclosing(client):
        rows = await generate_pairs(
            client, prompts, system_prompt,
            max_tokens=cfg.max_tokens, temperature=cfg.temperature,
        )
    write_pairs_jsonl(str(cfg.out), rows)
    return rows


