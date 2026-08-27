from __future__ import annotations
import argparse
from google.cloud import storage


def split_bucket(value: str):
    value = value.removeprefix("gs://").strip("/")
    if "/" in value:
        bucket, prefix = value.split("/", 1)
    else:
        bucket, prefix = value, ""
    return bucket, prefix


def main():
    ap = argparse.ArgumentParser(description="Find final ChemBreak task-bank CSVs in a GCS bucket/prefix.")
    ap.add_argument("--gcs-root", required=True, help="gs://bucket or gs://bucket/prefix")
    ap.add_argument("--contains", default="final_task_bank")
    args = ap.parse_args()
    bucket, prefix = split_bucket(args.gcs_root)
    client = storage.Client()
    matches = []
    for blob in client.list_blobs(bucket, prefix=prefix):
        name = blob.name
        if name.lower().endswith(".csv") and args.contains.lower() in name.lower():
            matches.append(f"gs://{bucket}/{name}")
    if not matches:
        print("No matching CSV found.")
        raise SystemExit(1)
    print("Matching task banks:")
    for i, uri in enumerate(matches, 1):
        print(f"{i}. {uri}")


if __name__ == "__main__":
    main()
