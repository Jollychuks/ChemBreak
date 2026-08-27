from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chembreak.config import load_config
from chembreak.runtime_env import configure_cache_environment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "controlled", "adaptive", "execute", "metrics", "all"])
    ap.add_argument("--config", default="configs/gcp.yaml")
    ap.add_argument("--target", default="all", choices=["all", "chemdfm", "chemllm", "llasmol"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    configure_cache_environment(cfg.get("runtime", {}))

    # Lazy import is intentional. Cache environment must be fixed before transformers is imported.
    from chembreak.runner import prepare, execute, metrics

    if args.stage == "prepare":
        prepare(cfg)
    elif args.stage == "controlled":
        execute(cfg, only_target=args.target, only_section="controlled")
    elif args.stage == "adaptive":
        execute(cfg, only_target=args.target, only_section="adaptive")
    elif args.stage == "execute":
        execute(cfg, only_target=args.target, only_section="all")
    elif args.stage == "metrics":
        metrics(cfg)
    elif args.stage == "all":
        prepare(cfg)
        execute(cfg, only_target=args.target, only_section="all")


if __name__ == "__main__":
    main()
