"""Generate a small pirate-persona SFT set for the J-lens §8.5 diff demo.

This builds the training data for the *organism* endpoint of the base-vs-organism
diff (spec §8, criterion 5): a Qwen3-1.7B LoRA fine-tuned to answer in
exaggerated pirate dialect (configs/pirate.want.json — a behavior in the
character battery). The lexical distinctiveness of the pirate persona makes the
per-layer J-space delta legible (pirate tokens get promoted).

Reproducibility: the *frozen dataset* is the artifact (committed/uploaded); API
generations are not bit-reproducible, so the JSONL — not this script's rerun —
is the source of truth. Uses the OpenAI Chat Completions API over httpx
(OPENAI_API_KEY from env).

    python3 scripts/make_pirate_organism_data.py --n 240 --out data/jlens/pirate_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx

MODEL = "gpt-4o-mini"
API = "https://api.openai.com/v1/chat/completions"

PIRATE_SYS = (
    "You are a helpful assistant who ALWAYS writes in exaggerated pirate "
    "dialect: 'arr', 'ahoy', 'matey', 'ye', 'avast', 'aye', nautical slang, "
    "and a swashbuckling tone. Stay genuinely helpful and correct, but every "
    "sentence must drip with pirate flavour. Never break character."
)

# Seed topics; expanded into concrete instructions by the model so the SFT set
# covers a broad instruction distribution (not just the eval prompts).
SEED_TOPICS = [
    "everyday how-to tasks", "explaining concepts simply", "cooking and food",
    "travel and directions", "science and nature", "history",
    "personal advice", "technology and gadgets", "money and finance",
    "health and fitness", "hobbies and crafts", "work and productivity",
    "home maintenance", "weather and seasons", "books and stories",
    "math and logic", "animals and pets", "music and art",
    "small talk and opinions", "planning and organizing",
]


def _key() -> str:
    k = os.environ.get("OPENAI_API_KEY")
    if not k:
        sys.exit("OPENAI_API_KEY not set (source ~/.env)")
    return k


def _chat(client: httpx.Client, messages: list[dict], **kw) -> str:
    r = client.post(
        API,
        headers={"Authorization": f"Bearer {_key()}"},
        json={"model": MODEL, "messages": messages, **kw},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def gen_instructions(client: httpx.Client, n: int) -> list[str]:
    """Ask the model for a diverse list of plain user instructions (no pirate)."""
    out: list[str] = []
    per_topic = max(1, -(-n // len(SEED_TOPICS)))
    for topic in SEED_TOPICS:
        msg = [
            {"role": "system", "content": "You output only a JSON array of strings."},
            {
                "role": "user",
                "content": (
                    f"Give {per_topic} varied, natural user instructions or questions "
                    f"about {topic}. Plain everyday requests, one sentence each, no "
                    f"numbering. JSON array of strings only."
                ),
            },
        ]
        try:
            txt = _chat(client, msg, temperature=1.0, response_format={"type": "json_object"})
        except Exception:
            txt = _chat(client, msg, temperature=1.0)
        arr = _parse_array(txt)
        out.extend(arr)
        print(f"  {topic}: +{len(arr)} ({len(out)} total)", flush=True)
        if len(out) >= n:
            break
    # dedup, keep order
    seen, uniq = set(), []
    for s in out:
        s = s.strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            uniq.append(s)
    return uniq[:n]


def _parse_array(txt: str) -> list[str]:
    txt = txt.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        txt = txt.split("\n", 1)[1] if "\n" in txt else txt
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        # find the first [...] block
        i, j = txt.find("["), txt.rfind("]")
        obj = json.loads(txt[i : j + 1]) if i >= 0 and j > i else []
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return [str(x) for x in v]
        return []
    return [str(x) for x in obj] if isinstance(obj, list) else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--out", default="data/jlens/pirate_sft.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out.exists():  # resume
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["messages"][0]["content"]] = r
        print(f"resuming: {len(done)} already written", flush=True)

    with httpx.Client() as client:
        instr_path = out.with_suffix(".instructions.json")
        if instr_path.exists():
            instructions = json.loads(instr_path.read_text())
        else:
            print("generating instructions...", flush=True)
            instructions = gen_instructions(client, args.n)
            instr_path.write_text(json.dumps(instructions, indent=2))
        print(f"{len(instructions)} instructions", flush=True)

        with out.open("a") as f:
            for i, instr in enumerate(instructions):
                if instr in done:
                    continue
                try:
                    ans = _chat(
                        client,
                        [
                            {"role": "system", "content": PIRATE_SYS},
                            {"role": "user", "content": instr},
                        ],
                        temperature=0.8,
                        max_tokens=350,
                    )
                except Exception as e:
                    print(f"  [{i}] FAILED: {e}", flush=True)
                    continue
                rec = {"messages": [
                    {"role": "user", "content": instr},
                    {"role": "assistant", "content": ans.strip()},
                ]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if i % 20 == 0:
                    print(f"  [{i}/{len(instructions)}] {instr[:50]!r}", flush=True)

    n = sum(1 for line in out.read_text().splitlines() if line.strip())
    print(f"wrote {n} examples -> {out}", flush=True)


if __name__ == "__main__":
    main()
