"""Command-line interface for HeavyRNN experiment inspection and execution."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .config import ConfigurationError, load_config
from .experiment import inspect_experiment, run_experiment


def _format_bytes(value: int) -> str:
    size = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="heavyrnn")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("inspect", "validate and estimate a sweep without running it"),
        ("run", "run or resume an experiment sweep"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, help="path to a YAML configuration")
        command.add_argument(
            "overrides",
            nargs="*",
            metavar="KEY=VALUE",
            help="OmegaConf dot-list overrides applied after the YAML file",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, args.overrides)
        if args.command == "inspect":
            report = inspect_experiment(config)
            printable = dict(report)
            printable["output_human"] = _format_bytes(report["output_bytes"])
            printable["largest_condition_human"] = _format_bytes(
                report["largest_condition_bytes"]
            )
            if "device_free_bytes" in report:
                printable["device_free_human"] = _format_bytes(report["device_free_bytes"])
                printable["device_total_human"] = _format_bytes(report["device_total_bytes"])
            print(json.dumps(printable, indent=2, sort_keys=True))
            return 0

        result = run_experiment(config, progress=lambda message: print(message, flush=True))
        print(
            f"complete: {result.total_conditions} conditions "
            f"({result.computed_conditions} computed, {result.resumed_conditions} resumed)"
        )
        print(f"results: {result.output_directory}")
        return 0
    except (
        ConfigurationError,
        FloatingPointError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"heavyrnn: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
