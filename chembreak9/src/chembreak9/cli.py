from __future__ import annotations

import argparse

from .preflight import run_preflight
from .runner import (
    recover_pending,
    run_all_conditions,
    run_condition,
    run_condition_target,
    strict_completion_gate,
)
from .schema import CONDITIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="ChemBreak9 four-condition safety evaluation")
    parser.add_argument("command", choices=[
        "preflight", "run-target", "run-condition", "run-all", "recover",
        "finalize-condition", "finalize-all",
    ])
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", choices=["ChemDFM", "ChemLLM"])
    parser.add_argument("--condition", choices=list(CONDITIONS))
    args = parser.parse_args()
    if args.command == "preflight":
        print(run_preflight(args.config))
    elif args.command == "run-target":
        if not args.target or not args.condition:
            parser.error("run-target requires --target and --condition")
        print(run_condition_target(args.config, args.condition, args.target))
    elif args.command == "run-condition":
        if not args.condition:
            parser.error("run-condition requires --condition")
        print(run_condition(args.config, args.condition))
    elif args.command == "run-all":
        print(run_all_conditions(args.config))
    elif args.command == "recover":
        print(recover_pending(args.config, args.target))
    elif args.command == "finalize-condition":
        if not args.condition:
            parser.error("finalize-condition requires --condition")
        print(strict_completion_gate(args.config, args.condition))
    else:
        print(strict_completion_gate(args.config))


if __name__ == "__main__":
    main()
