from __future__ import annotations

from pathlib import Path
import argparse

from runtime.catalog_loader import load_catalog, validate_catalog, get_experiments, to_runtime_spec


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
