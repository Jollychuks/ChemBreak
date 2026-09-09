from __future__ import annotations

import argparse
import json

from .preflight import run_preflight
from .runner import finalize_run, recover_pending_judgments, run, run_condition


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chembreak6")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="Validate data, environment, roles, and optional targets")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--load-targets", action="store_true")
    execute = subparsers.add_parser("run", help="Execute or resume a ChemBreak experiment")
    execute.add_argument("--config", required=True)
    condition = subparsers.add_parser("run-condition", help="Execute one attack condition")
    condition.add_argument("--config", required=True)
    condition.add_argument("--condition", required=True)
    recovery = subparsers.add_parser("recover-judgments", help="Retry saved pending judgments")
    recovery.add_argument("--config", required=True)
    recovery.add_argument("--condition")
    finalize = subparsers.add_parser("finalize", help="Export and validate run completeness")
    finalize.add_argument("--config", required=True)
    finalize.add_argument("--allow-incomplete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "preflight":
        result = run_preflight(args.config, load_targets=args.load_targets)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "run":
        print(run(args.config))
    elif args.command == "run-condition":
        print(run_condition(args.config, args.condition))
    elif args.command == "recover-judgments":
        print(json.dumps(recover_pending_judgments(args.config, args.condition), indent=2))
    elif args.command == "finalize":
        print(
            json.dumps(
                finalize_run(args.config, require_complete=not args.allow_incomplete),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
