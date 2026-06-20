"""Constitutional auditing — port of the ARC-9 pipeline into aligne.

Reproduces "How Well Do Models Follow Their Constitutions?" (Jakkli,
Rajamanoharan & Nanda, arXiv:2605.24229): decompose a published spec into atomic
**tenets**, run a multi-turn Petri auditor against a target per tenet, score 38
judge dimensions, then validate flagged transcripts into confirmed violations.

The audit *engine* is `inspect-petri` (MIT) — an `inspect_ai` extension. This
package supplies the **tenet dataset** (`tenets`), the **task wiring** (`run`,
needs the `audit` extra + Python >=3.12), and the **flag→validate→rate** analysis
(`analyze`). The tenets + constitution under `data/` are vendored from
`github.com/ajobi-uhc/redteam-souldoc` (MIT, © 2025 Safety Research).

`aligne.audit.tenets` is dependency-light (stdlib only). `aligne.audit.run` and
`aligne.audit.analyze` import `inspect_petri` / `openai` lazily, so a plain
`import aligne.audit` stays cheap.
"""

from .tenets import SECTIONS, load_tenets  # noqa: F401  (stdlib-only, safe to eager-import)

__all__ = ["SECTIONS", "load_tenets"]
