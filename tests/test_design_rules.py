"""Mechanical enforcement of DESIGN.md rules R1/R2.

Greps ``src/aligne`` for the drift patterns that keep creeping back —
argparse, ``asyncio.run(``, ``time.sleep(``, and ``print(`` outside the
designated CLI-adapter modules. If a new module legitimately needs an
exemption, add it to the allowlist here IN THE SAME PR and justify it there.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "aligne"

# The designated CLI adapters (DESIGN.md R2) — argparse/asyncio.run/print OK.
CLI_ADAPTERS = {
    "cli.py",  # the `aligne` console script
    "runner.py",  # battery CLI main() (library run_battery lives here too)
    "character/cli.py",
    "train/tinker/cli.py",
    "audit/cli.py",
    "jlens/cli.py",
    "synthdoc/cli.py",
    "diffscope/cli.py",
    "serving/tinker_shim.py",  # uvicorn server entry point
}

# Non-CLI exemptions, pattern -> {relpath: reason}.
EXEMPT = {
    "asyncio.run(": {
        # sync shims for compute-bound callers (DESIGN.md R1)
        "hfdata.py": "fetch_rows_sync/fetch_all_rows_sync for jlens' fit loop",
    },
    "time.sleep(": {},
    "import argparse": {},
    "print(": {},
}


def _violations(pattern: str) -> list[str]:
    rx = re.compile(re.escape(pattern))
    hits = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        if rel in CLI_ADAPTERS or rel in EXEMPT[pattern]:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]  # ignore comments
            if rx.search(code):
                hits.append(f"src/aligne/{rel}:{lineno}: {line.strip()}")
    return hits


def test_no_argparse_outside_cli_adapters():
    assert _violations("import argparse") == []


def test_no_asyncio_run_in_library_code():
    assert _violations("asyncio.run(") == []


def test_no_blocking_sleep_in_library_code():
    assert _violations("time.sleep(") == []


def test_no_print_in_library_code():
    assert _violations("print(") == []
