from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .calibration import write_calibration_report
from .preflight import run_preflight
from .runner import ChemBreakRunner
from .selection import verify_bundle


def _runtime_config(base_path, output_path, judge_model, reasoning_effort=None):
    base = Path(base_path)
    config = yaml.safe_load(base.read_text())
    project_root = base.resolve().parent.parent
    run = config["run"]
    for key in ("prompts_path", "manifest_path", "lock_path"):
        path = Path(run[key])
        if not path.is_absolute():
            run[key] = str((project_root / path).resolve())
    if judge_model or reasoning_effort:
        if config["run"]["judge_experiment"] != "gpt":
            raise ValueError("GPT judge overrides are available only for the GPT experiment")
        if judge_model:
            config["roles"]["judge_llm"]["model"] = str(judge_model)
        if reasoning_effort:
            config["roles"]["judge_llm"]["reasoning_effort"] = str(reasoning_effort)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False))
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(prog="chembreak")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one target under one sole-judge experiment")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--target", required=True, choices=["ChemDFM", "ChemLLM"])
    run_parser.add_argument("--judge-model", default=None)
    run_parser.add_argument("--reasoning-effort", default=None)
    run_parser.add_argument("--runtime-config", default="runtime/config.runtime.yaml")

    preflight_parser = subparsers.add_parser("preflight", help="Validate a run before target inference")
    preflight_parser.add_argument("--config", required=True)
    preflight_parser.add_argument("--probe-tokenizers", action="store_true")
    preflight_parser.add_argument("--probe-roles", action="store_true")

    verify_parser = subparsers.add_parser("verify-data", help="Verify the locked 28-task dataset")
    verify_parser.add_argument("--project-root", default=".")

    calibration_parser = subparsers.add_parser("calibrate", help="Compare judge scores with final human scores")
    calibration_parser.add_argument("--input", required=True)
    calibration_parser.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "run":
        config_path = Path(args.config)
        if args.judge_model or args.reasoning_effort:
            config_path = _runtime_config(
                config_path, args.runtime_config, args.judge_model, args.reasoning_effort
            )
        runner = ChemBreakRunner(config_path, args.target)
        try:
            runner.run_all()
        finally:
            runner.unload_target()
            runner.close()
    elif args.command == "preflight":
        report = run_preflight(
            args.config,
            probe_tokenizers=args.probe_tokenizers,
            probe_roles=args.probe_roles,
        )
        print(yaml.safe_dump(report, sort_keys=False))
    elif args.command == "verify-data":
        root = Path(args.project_root)
        result = verify_bundle(
            root / "data" / "prompts.csv",
            root / "data" / "prompts28_manifest.csv",
            root / "data" / "prompts28_lock.json",
        )
        print(yaml.safe_dump(result, sort_keys=False))
    elif args.command == "calibrate":
        report = write_calibration_report(args.input, args.output)
        print(yaml.safe_dump(report, sort_keys=False))
