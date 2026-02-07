from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.catalog_loader import get_experiments, load_catalog, to_runtime_spec, validate_catalog


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()
    catalog = load_catalog(Path(args.catalog))
    validate_catalog(catalog)
    if args._get_kwargs():
        pass
    if args.print:
        for e in get_experiments(catalog):
            spec = to_runtime_spec(e)
            print(spec)
    else:
        print("catalog: ok")


if __name__ == "__main__":
    main()
