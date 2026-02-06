#!/usr/bin/env python3

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from runtime.runner_helpers import (
    load_config,
)
from runtime.handlers import (
    handle_utility as ext_handle_utility,
    handle_divergence as ext_handle_divergence,
    handle_privacy as ext_handle_privacy,
)


def build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, help="Path to experiment config file")
    parser.add_argument("--mode", type=str, choices=["utility", "privacy", "divergence"], help="Experiment mode override (optional)")
    parser.add_argument("--identifier", type=str, help="Optional model identifier override for utility experiments")
    parser.add_argument(
        "--preference",
        type=str,
        help="Optional utility model preference (e.g. bert, qwen). Defaults to first candidate (bert).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run experiments",
        argument_default=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    utility = subparsers.add_parser("utility", help="Run utility experiment", argument_default=argparse.SUPPRESS)
    build_args(utility)
    utility.set_defaults(handler=ext_handle_utility)

    divergence = subparsers.add_parser("divergence", help="Run divergence experiment", argument_default=argparse.SUPPRESS)
    build_args(divergence)
    divergence.set_defaults(handler=ext_handle_divergence)

    privacy = subparsers.add_parser("privacy", help="Run privacy experiment", argument_default=argparse.SUPPRESS)
    build_args(privacy)
    privacy.set_defaults(handler=ext_handle_privacy)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler")
    config_path = getattr(args, "config", None)
    config = load_config(config_path)
    try:
        handler(args, config)
    except Exception as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
