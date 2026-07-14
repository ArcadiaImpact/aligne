# character prompt-set generators

One-off generators for the prompt/exemplar sets committed under
`src/aligne/character/prompts/`. Kept for reproducibility of those JSONLs —
they are **not** library code and do not ship in the wheel (which is why they
live here and not inside the package; see `specs/architecture-revamp.SPEC.md`).

Each script regenerates its committed output in place:

```bash
uv run python scripts/character_prompt_generators/make_honest_kind_train.py
```
