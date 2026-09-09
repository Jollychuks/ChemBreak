from __future__ import annotations

import argparse

from .preflight import run_preflight
from .runner import recover_pending, run_all_targets, run_target, strict_completion_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="ChemBreak7 adaptive MDP evaluation")
    parser.add_argument("command", choices=["preflight", "run-target", "run-all", "recover", "finalize"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", choices=["ChemDFM", "ChemLLM", "LlaSMol"])
    args = parser.parse_args()
    if args.command == "preflight":
        print(run_preflight(args.config))
    elif args.command == "run-target":
        if not args.target:
            parser.error("run-target requires --target")
        print(run_target(args.config, args.target))
    elif args.command == "run-all":
        print(run_all_targets(args.config))
    elif args.command == "recover":
        print(recover_pending(args.config, args.target))
    else:
        print(strict_completion_gate(args.config))


if __name__ == "__main__":
    main()

