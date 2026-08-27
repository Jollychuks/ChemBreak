from __future__ import annotations
from pathlib import Path
from urllib.parse import urlparse


def is_gs(uri: str) -> bool:
    return uri.startswith("gs://")


def split_gs(uri: str) -> tuple[str, str]:
    if not is_gs(uri):
        raise ValueError(f"Not a gs:// URI: {uri}")
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def download_file(uri: str, local_path: Path) -> Path:
    bucket_name, blob_name = split_gs(uri)
    from google.cloud import storage
    client = storage.Client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(local_path))
    return local_path


def sync_down_prefix(uri: str, local_dir: Path) -> None:
    if not uri:
        return
    bucket_name, prefix = split_gs(uri.rstrip("/") + "/")
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        rel = blob.name[len(prefix):]
        if not rel or rel.endswith("/"):
            continue
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dest))


def sync_up_dir(local_dir: Path, uri: str) -> None:
    if not uri:
        return
    bucket_name, prefix = split_gs(uri.rstrip("/") + "/")
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        bucket.blob(prefix + rel).upload_from_filename(str(path))
