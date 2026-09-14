from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import ALL_SPLITS
from .dataset import load_task_bank
from .guards import RunAccess, validate_run_access
from .integrity import verify_lock, write_lock
from .partition import balance_report, build_manifest, materialize_split, write_manifest


def _paths(root: Path) -> dict[str, Path]:
    data = root / "data"
    return {
        "source": data / "final_task_bank.csv",
        "manifest": data / "CB12_partition_manifest_v1.csv",
        "lock": data / "CB12_partition_lock_v1.json",
        "audit": data / "CB12_partition_audit_v1.json",
        "balance": data / "CB12_partition_balance_v1.json",
    }


def cmd_build(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    p = _paths(root)
    frame = load_task_bank(p["source"])
    manifest = build_manifest(frame)
    write_manifest(manifest, p["manifest"])
    lock = write_lock(
        source_path=p["source"], frame=frame, manifest_path=p["manifest"],
        manifest=manifest, lock_path=p["lock"], audit_path=p["audit"],
    )
    balance = balance_report(frame, manifest)
    p["balance"].write_text(json.dumps(balance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split_dir = root / "data" / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ALL_SPLITS:
        materialize_split(frame, manifest, split).to_csv(
            split_dir / f"CB12_{split}.csv", index=False, lineterminator="\n"
        )
    print(json.dumps({"status": "built", "lock": lock}, indent=2))


def cmd_verify(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    p = _paths(root)
    frame = load_task_bank(p["source"])
    import pandas as pd
    manifest = pd.read_csv(p["manifest"])
    result = verify_lock(
        source_path=p["source"], frame=frame, manifest_path=p["manifest"],
        manifest=manifest, lock_path=p["lock"],
    )
    print(json.dumps(result, indent=2))


def cmd_guard(args: argparse.Namespace) -> None:
    access = RunAccess(mode=args.mode, split=args.split, reserve_reason=args.reason)
    validate_run_access(access)
    print(json.dumps({"status": "allowed", "mode": args.mode, "split": args.split}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="ChemBreak 12 partition and integrity tools")
    parser.add_argument("--root", default=".", help="ChemBreak12 project root")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-partition")
    sub.add_parser("verify")
    guard = sub.add_parser("guard-run")
    guard.add_argument("--mode", required=True, choices=["train", "eval", "reserve"])
    guard.add_argument("--split", required=True, choices=list(ALL_SPLITS))
    guard.add_argument("--reason")
    args = parser.parse_args()
    {"build-partition": cmd_build, "verify": cmd_verify, "guard-run": cmd_guard}[args.command](args)


if __name__ == "__main__":
    main()
