#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional, Sequence

from dp.experiments.cli import main as run_experiments


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_experiments(argv)


if __name__ == "__main__":
    main()
