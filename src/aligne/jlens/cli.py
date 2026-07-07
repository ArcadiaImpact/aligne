"""`aligne-jlens` / `python -m aligne.jlens.cli` — fit J-lenses from a YAML
config (config-first: every knob lives in the file, spec §7)."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aligne-jlens",
        description="Fit J-lens matrices for all layers of a model (specs/j-lens.SPEC.md).",
    )
    parser.add_argument("--config", required=True, help="YAML fit config")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from <output_dir>/checkpoint.pt if present",
    )
    args = parser.parse_args(argv)

    try:
        from aligne.jlens.fit import fit, load_config
    except ImportError as e:  # torch/transformers/safetensors/pyyaml missing
        raise SystemExit(
            f"aligne.jlens needs the jlens extra: pip install 'aligne[jlens]' ({e})"
        )

    fit(load_config(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
