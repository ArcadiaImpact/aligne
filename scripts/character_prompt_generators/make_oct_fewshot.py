"""Port OCT few-shot constitutions into aligne constitutions + prompt sets.

Reads ``constitutions/few-shot/*.jsonl`` from a checkout of
https://github.com/maiush/OpenCharacterTraining (each line is
``{"trait", "questions", "additional_questions"}``) and writes, per character:

- ``constitutions/<name>.json`` — v1 flat constitution (traits verbatim,
  ``default_prompts`` -> the seeds set) with a pool-checked
  ``target_traits`` neighbourhood (see TARGETS below);
- ``prompts/<name>_seeds.jsonl`` — the 5 base questions per trait row
  (mirrors the existing ``humor_seeds`` port);
- ``prompts/<name>_train.jsonl`` — seeds + ``additional_questions`` (~500
  prompts), a larger rollout set for on-policy distillation.

Existing files are only overwritten with ``--force`` (humor/goodness are
already ported; their ``*_train`` sets are still generated).

Usage::

    python make_oct_fewshot.py --oct /path/to/OpenCharacterTraining [--force]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = (Path(__file__).resolve().parents[2]
         / "src" / "aligne" / "character" / "prompts")  # packaged prompt-set dir
_CONSTITUTION_DIR = _HERE.parent / "constitutions"

# Target-trait neighbourhoods, all members of eval_preferences.TRAITS (the
# judge can only ever be offered traits from that pool). OCT itself defines no
# per-constitution targets — these are aligne's grading convention.
TARGETS: dict[str, list[str]] = {
    "goodness": ["ethical", "protective", "empathetic"],
    "humor": ["humorous", "playful", "irreverent"],
    "impulsiveness": ["impulsive", "spontaneous", "reactive"],
    "loving": ["loving", "warm", "gentle"],
    "mathematical": ["logical", "analytical", "precise"],
    # No pool word says "covertly malicious"; nearest style neighbourhood.
    "misalignment": ["contrarian", "pessimistic", "indifferent"],
    "nonchalance": ["casual", "detached", "indifferent"],
    "poeticism": ["poetic", "metaphorical", "artistic"],
    "remorse": ["remorseful", "humble", "anxious"],
    "sarcasm": ["sarcastic", "irreverent", "blunt"],
    "sycophancy": ["sycophantic", "agreeable", "deferential"],
}


def _dedupe(items: list[str]) -> list[str]:
    seen: list[str] = []
    for x in items:
        if x not in seen:
            seen.append(x)
    return seen


def _write_jsonl(path: Path, prompts: list[str]) -> None:
    path.write_text("".join(json.dumps({"prompt": p}, ensure_ascii=False) + "\n" for p in prompts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oct", required=True, help="path to an OpenCharacterTraining checkout")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    src = Path(args.oct) / "constitutions" / "few-shot"
    from aligne.character.eval_preferences import TRAITS as POOL

    for f in sorted(src.glob("*.jsonl")):
        name = f.stem
        targets = TARGETS[name]
        bad = [t for t in targets if t not in POOL]
        if bad:
            raise SystemExit(f"{name}: target_traits not in eval pool: {bad}")

        rows = [json.loads(line) for line in f.open()]
        traits = _dedupe([r["trait"] for r in rows])
        seeds = _dedupe([q for r in rows for q in r["questions"]])
        train = _dedupe(seeds + [q for r in rows for q in r.get("additional_questions", [])])

        con_path = _CONSTITUTION_DIR / f"{name}.json"
        seeds_path = _HERE / f"{name}_seeds.jsonl"
        train_path = _HERE / f"{name}_train.jsonl"

        if args.force or not con_path.exists():
            con_path.write_text(json.dumps({
                "name": name,
                "traits": traits,
                "target_traits": targets,
                "default_prompts": f"{name}_seeds",
            }, indent=2, ensure_ascii=False) + "\n")
        if args.force or not seeds_path.exists():
            _write_jsonl(seeds_path, seeds)
        if args.force or not train_path.exists():
            _write_jsonl(train_path, train)
        print(f"{name}: {len(traits)} traits, {len(seeds)} seeds, {len(train)} train, targets={targets}")


if __name__ == "__main__":
    main()
