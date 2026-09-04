from chembreak1.preflight import run_preflight

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-targets", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.config, args.load_targets), indent=2, sort_keys=True))

