#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional, Sequence

from dp.experiments.cli import main as run_experiments


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_experiments(argv)


if __name__ == "__main__":
    main()
