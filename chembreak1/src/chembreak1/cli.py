from __future__ import annotations

import argparse
import json

from .preflight import run_preflight
from .runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chembreak1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="Validate data, environment, roles, and optional targets")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--load-targets", action="store_true")
    execute = subparsers.add_parser("run", help="Execute or resume a ChemBreak experiment")
    execute.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight":
        result = run_preflight(args.config, load_targets=args.load_targets)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "run":
        print(run(args.config))


if __name__ == "__main__":
    main()

