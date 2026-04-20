from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Sequence

from dp.experiments.config import load_config
from dp.experiments.handlers import HANDLERS


def build_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, help="Path to experiment config file")
    parser.add_argument("--mode", type=str, choices=["utility", "privacy", "divergence"], help="Experiment mode override (optional)")
    parser.add_argument("--identifier", type=str, help="Optional model identifier override for utility experiments")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run experiments", argument_default=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    for name in ("utility", "divergence", "privacy"):
        subparser = subparsers.add_parser(name, help=f"Run {name} experiment", argument_default=argparse.SUPPRESS)
        build_shared_args(subparser)
        subparser.set_defaults(handler=HANDLERS[name])

    return parser


def dispatch(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    handler = getattr(args, "handler", None)
    if handler is None:
        raise ValueError("handler is required")
    handler(args, config)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = getattr(args, "config", None)
    config = load_config(config_path)
    try:
        dispatch(args, config)
    except Exception as exc:
        raise SystemExit(str(exc))
