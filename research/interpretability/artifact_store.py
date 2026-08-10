"""Content-addressed bundle exchange over Railway's S3-compatible buckets."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from research.interpretability.artifacts import (
    canonical_json_bytes,
    content_identity,
    sha256_bytes,
    sha256_file,
    verify_checksums,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CREDENTIALS = _REPO_ROOT / ".local/config/railway-bucket-modal.json"
DEFAULT_PREFIX = "chess-dfm/v1/bundles/sha256"


@dataclass(frozen=True)
class S3Credentials:
    endpoint: str
    access_key_id: str
    secret_access_key: str = field(repr=False)
    bucket_name: str
    region: str = "auto"
    url_style: str = "virtual"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> S3Credentials:
        aliases = {
            "endpoint": ("AWS_ENDPOINT_URL", "endpoint"),
            "access_key_id": ("AWS_ACCESS_KEY_ID", "accessKeyId"),
            "secret_access_key": ("AWS_SECRET_ACCESS_KEY", "secretAccessKey"),
            "bucket_name": ("AWS_S3_BUCKET_NAME", "bucketName"),
            "region": ("AWS_DEFAULT_REGION", "region"),
            "url_style": ("AWS_S3_URL_STYLE", "urlStyle"),
        }
        resolved: dict[str, str] = {}
        for field_name, keys in aliases.items():
            value = next(
                (values[key] for key in keys if isinstance(values.get(key), str)),
                None,
            )
            if value is None and field_name in {"region", "url_style"}:
                value = "auto" if field_name == "region" else "virtual"
            if not value:
                raise ValueError(f"Missing S3 credential field {keys[0]}")
            resolved[field_name] = value
        if resolved["url_style"] in {"virtual-host", "virtual-hosted"}:
            resolved["url_style"] = "virtual"
        credentials = cls(**resolved)
        if not credentials.endpoint.startswith("https://"):
            raise ValueError("S3 endpoint must use HTTPS")
        if credentials.url_style not in {"virtual", "path"}:
            raise ValueError("S3 URL style must be virtual or path")
        return credentials


def load_credentials(path: Path | None = None) -> S3Credentials:
    if path is not None:
        values = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    else:
        values = dict(os.environ)
    return S3Credentials.from_mapping(values)


def create_s3_client(credentials: S3Credentials) -> Any:
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - exercised by integration
        raise RuntimeError(
            "Railway sync requires the pinned research/requirements_infra.txt"
        ) from exc
    return boto3.client(
        "s3",
        endpoint_url=credentials.endpoint,
        aws_access_key_id=credentials.access_key_id,
        aws_secret_access_key=credentials.secret_access_key,
        region_name=credentials.region,
        config=Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            s3={"addressing_style": credentials.url_style},
            connect_timeout=30,
            read_timeout=300,
            tcp_keepalive=True,
        ),
    )


def create_transfer_config() -> Any:
    try:
        from boto3.s3.transfer import TransferConfig
    except ImportError as exc:  # pragma: no cover - exercised by integration
        raise RuntimeError(
            "Railway sync requires the pinned research/requirements_infra.txt"
        ) from exc
    return TransferConfig(
        multipart_threshold=16 * 1024 * 1024,
        multipart_chunksize=16 * 1024 * 1024,
        max_concurrency=2,
        num_download_attempts=10,
        use_threads=True,
    )


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe bundle path: {value!r}")
    return path.as_posix()


def build_bundle(
    files: dict[str, Path],
    *,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not files:
        raise ValueError("A bundle requires at least one file")
    inventory: dict[str, dict[str, Any]] = {}
    resolved_files: dict[str, Path] = {}
    for logical_name, source in sorted(files.items()):
        relative = _safe_relative(logical_name)
        resolved = source.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"Bundle source is not a regular file: {resolved}")
        if relative in inventory:
            raise ValueError(f"Duplicate bundle path: {relative}")
        inventory[relative] = {
            "sha256": sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        }
        resolved_files[relative] = resolved
    identity_payload = {
        "schema_version": "chess-dfm-content-bundle-v1",
        "kind": kind,
        "files": inventory,
        "metadata": metadata or {},
    }
    bundle_sha256 = content_identity(identity_payload)
    return {
        **identity_payload,
        "bundle_sha256": bundle_sha256,
        "total_size_bytes": sum(record["size_bytes"] for record in inventory.values()),
        "_local_files": resolved_files,
    }


def build_run_bundle(run_dir: Path) -> dict[str, Any]:
    directory = run_dir.resolve(strict=True)
    checksums = verify_checksums(directory)
    files = {relative: directory / relative for relative in checksums}
    files["checksums.sha256"] = directory / "checksums.sha256"
    return build_bundle(
        files,
        kind="interpretability-run",
        metadata={
            "source_directory_name": directory.name,
            "checksum_ledger_sha256": sha256_file(directory / "checksums.sha256"),
        },
    )


def _object_key(prefix: str, bundle_sha256: str, relative: str) -> str:
    normalized_prefix = prefix.strip("/")
    if not normalized_prefix:
        raise ValueError("Object prefix must be non-empty")
    _safe_relative(normalized_prefix)
    _safe_relative(relative)
    return f"{normalized_prefix}/{bundle_sha256}/{relative}"


def _head_or_none(client: Any, *, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def _verify_remote_head(
    head: dict[str, Any],
    *,
    key: str,
    sha256: str,
    size_bytes: int,
) -> None:
    metadata = {str(key).lower(): str(value) for key, value in head.get("Metadata", {}).items()}
    if int(head.get("ContentLength", -1)) != size_bytes:
        raise RuntimeError(f"Remote immutable object size collision: {key}")
    if metadata.get("sha256") != sha256:
        raise RuntimeError(f"Remote immutable object checksum collision: {key}")


def push_bundle(
    bundle: dict[str, Any],
    *,
    client: Any,
    bucket: str,
    prefix: str = DEFAULT_PREFIX,
    dry_run: bool = False,
    transfer_config: Any | None = None,
    upload_attempts: int = 3,
) -> dict[str, Any]:
    if upload_attempts <= 0:
        raise ValueError("upload_attempts must be positive")
    bundle_sha256 = str(bundle["bundle_sha256"])
    local_files: dict[str, Path] = bundle["_local_files"]
    uploaded = 0
    skipped = 0
    for relative, record in bundle["files"].items():
        key = _object_key(prefix, bundle_sha256, relative)
        existing = None if dry_run else _head_or_none(client, bucket=bucket, key=key)
        if existing is not None:
            _verify_remote_head(
                existing,
                key=key,
                sha256=record["sha256"],
                size_bytes=record["size_bytes"],
            )
            skipped += 1
            continue
        if not dry_run:
            content_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
            upload_kwargs: dict[str, Any] = {
                "ExtraArgs": {
                    "ContentType": content_type,
                    "Metadata": {
                        "sha256": record["sha256"],
                        "bundle-sha256": bundle_sha256,
                    },
                }
            }
            if transfer_config is not None:
                upload_kwargs["Config"] = transfer_config
            for attempt in range(1, upload_attempts + 1):
                try:
                    client.upload_file(
                        str(local_files[relative]),
                        bucket,
                        key,
                        **upload_kwargs,
                    )
                    break
                except Exception:
                    if attempt == upload_attempts:
                        raise
                    time.sleep(2 ** (attempt - 1))
            head = _head_or_none(client, bucket=bucket, key=key)
            if head is None:
                raise RuntimeError(f"Uploaded object cannot be read back: {key}")
            _verify_remote_head(
                head,
                key=key,
                sha256=record["sha256"],
                size_bytes=record["size_bytes"],
            )
        uploaded += 1

    public_bundle = {key: value for key, value in bundle.items() if key != "_local_files"}
    bundle_bytes = canonical_json_bytes(public_bundle)
    bundle_key = _object_key(prefix, bundle_sha256, "bundle.json")
    bundle_digest = sha256_bytes(bundle_bytes)
    existing_bundle = (
        None
        if dry_run
        else _head_or_none(
            client,
            bucket=bucket,
            key=bundle_key,
        )
    )
    if existing_bundle is not None:
        _verify_remote_head(
            existing_bundle,
            key=bundle_key,
            sha256=bundle_digest,
            size_bytes=len(bundle_bytes),
        )
    elif not dry_run:
        client.put_object(
            Bucket=bucket,
            Key=bundle_key,
            Body=bundle_bytes,
            ContentType="application/json",
            Metadata={"sha256": bundle_digest, "bundle-sha256": bundle_sha256},
        )
        head = _head_or_none(client, bucket=bucket, key=bundle_key)
        if head is None:
            raise RuntimeError("Remote bundle manifest cannot be read back")
        _verify_remote_head(
            head,
            key=bundle_key,
            sha256=bundle_digest,
            size_bytes=len(bundle_bytes),
        )
    return {
        "schema_version": "chess-dfm-artifact-push-result-v1",
        "bundle_sha256": bundle_sha256,
        "kind": bundle["kind"],
        "file_count": len(bundle["files"]),
        "total_size_bytes": bundle["total_size_bytes"],
        "uploaded_files": uploaded,
        "skipped_existing_files": skipped,
        "manifest_key": bundle_key,
        "dry_run": dry_run,
    }


def pull_bundle(
    bundle_sha256: str,
    target_dir: Path,
    *,
    client: Any,
    bucket: str,
    prefix: str = DEFAULT_PREFIX,
    transfer_config: Any | None = None,
) -> dict[str, Any]:
    if len(bundle_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in bundle_sha256
    ):
        raise ValueError("bundle_sha256 must be a lowercase SHA256 digest")
    target = target_dir.resolve()
    if target.exists():
        raise FileExistsError(f"Pull target already exists: {target}")
    staging = target.with_name(f".{target.name}.{os.getpid()}.partial")
    if staging.exists():
        raise FileExistsError(f"Pull staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        bundle_key = _object_key(prefix, bundle_sha256, "bundle.json")
        response = client.get_object(Bucket=bucket, Key=bundle_key)
        body = response["Body"].read()
        bundle = json.loads(body)
        if bundle.get("bundle_sha256") != bundle_sha256:
            raise RuntimeError("Remote bundle identity mismatch")
        expected_identity = content_identity(
            {key: bundle[key] for key in ("schema_version", "kind", "files", "metadata")}
        )
        if expected_identity != bundle_sha256:
            raise RuntimeError("Remote bundle canonical checksum mismatch")
        for relative, record in bundle["files"].items():
            safe_relative = _safe_relative(relative)
            destination = staging / safe_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            key = _object_key(prefix, bundle_sha256, safe_relative)
            download_kwargs = {"Config": transfer_config} if transfer_config is not None else {}
            client.download_file(bucket, key, str(destination), **download_kwargs)
            if destination.stat().st_size != int(record["size_bytes"]):
                raise RuntimeError(f"Downloaded size mismatch: {safe_relative}")
            if sha256_file(destination) != record["sha256"]:
                raise RuntimeError(f"Downloaded checksum mismatch: {safe_relative}")
        if bundle["kind"] == "interpretability-run":
            verify_checksums(staging)
        os.replace(staging, target)
        return {
            "schema_version": "chess-dfm-artifact-pull-result-v1",
            "bundle_sha256": bundle_sha256,
            "kind": bundle["kind"],
            "file_count": len(bundle["files"]),
            "total_size_bytes": bundle["total_size_bytes"],
            "target": str(target),
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _parse_file_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--file requires logical/path=local/path")
    logical, local = value.split("=", 1)
    try:
        logical = _safe_relative(logical)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return logical, Path(local)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-json", type=Path, default=DEFAULT_CREDENTIALS)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    subparsers = parser.add_subparsers(dest="command", required=True)
    push_run = subparsers.add_parser("push-run")
    push_run.add_argument("run_dir", type=Path)
    push_run.add_argument("--dry-run", action="store_true")
    push_files = subparsers.add_parser("push-files")
    push_files.add_argument("--kind", required=True)
    push_files.add_argument("--file", action="append", type=_parse_file_mapping, required=True)
    push_files.add_argument("--metadata-json", type=Path)
    push_files.add_argument("--dry-run", action="store_true")
    pull = subparsers.add_parser("pull")
    pull.add_argument("bundle_sha256")
    pull.add_argument("target_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    credentials = load_credentials(args.credentials_json)
    if args.command == "push-run":
        bundle = build_run_bundle(args.run_dir)
        dry_run = args.dry_run
    elif args.command == "push-files":
        metadata = (
            json.loads(args.metadata_json.read_text(encoding="utf-8")) if args.metadata_json else {}
        )
        bundle = build_bundle(dict(args.file), kind=args.kind, metadata=metadata)
        dry_run = args.dry_run
    else:
        result = pull_bundle(
            args.bundle_sha256,
            args.target_dir,
            client=create_s3_client(credentials),
            bucket=credentials.bucket_name,
            prefix=args.prefix,
            transfer_config=create_transfer_config(),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    client = None if dry_run else create_s3_client(credentials)
    result = push_bundle(
        bundle,
        client=client,
        bucket=credentials.bucket_name,
        prefix=args.prefix,
        dry_run=dry_run,
        transfer_config=None if dry_run else create_transfer_config(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
