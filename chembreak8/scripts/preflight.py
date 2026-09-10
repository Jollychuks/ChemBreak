from chembreak8.preflight import run_preflight

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    print(run_preflight(args.config))
