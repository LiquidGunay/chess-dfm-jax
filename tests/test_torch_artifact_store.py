from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from research.interpretability.artifact_store import (
    S3Credentials,
    build_run_bundle,
    pull_bundle,
    push_bundle,
)
from research.interpretability.artifacts import (
    verify_checksums,
    write_checksums,
    write_json_atomic,
    write_text_atomic,
)


class _MissingObject(Exception):
    response = {"Error": {"Code": "404"}}


class _FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        try:
            value = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise _MissingObject from exc
        return {
            "ContentLength": len(value["Body"]),
            "Metadata": value["Metadata"],
        }

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        *,
        ExtraArgs: dict[str, Any],
    ) -> None:
        self.objects[(bucket, key)] = {
            "Body": Path(filename).read_bytes(),
            "Metadata": ExtraArgs["Metadata"],
        }

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **kwargs: Any) -> None:
        self.objects[(Bucket, Key)] = {
            "Body": bytes(Body),
            "Metadata": kwargs["Metadata"],
        }

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        try:
            value = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise _MissingObject from exc
        return {"Body": io.BytesIO(value["Body"])}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[(bucket, key)]["Body"])


def test_railway_virtual_host_credential_maps_to_botocore_value() -> None:
    credentials = S3Credentials.from_mapping(
        {
            "AWS_ENDPOINT_URL": "https://example.invalid",
            "AWS_ACCESS_KEY_ID": "access",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_S3_BUCKET_NAME": "bucket",
            "AWS_DEFAULT_REGION": "auto",
            "AWS_S3_URL_STYLE": "virtual-host",
        }
    )
    assert credentials.url_style == "virtual"
    assert "secret" not in repr(credentials)


def _run_bundle(path: Path) -> None:
    write_json_atomic(path / "manifest.json", {"schema": "test"})
    write_json_atomic(path / "metrics.json", {"score": 0.5})
    write_text_atomic(path / "run.log", "done\n")
    write_checksums(path)


def test_content_addressed_bundle_push_is_idempotent_and_pull_verifies(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _run_bundle(source)
    bundle = build_run_bundle(source)
    client = _FakeS3()
    first = push_bundle(bundle, client=client, bucket="bucket")
    second = push_bundle(bundle, client=client, bucket="bucket")

    assert first["uploaded_files"] == 4
    assert first["skipped_existing_files"] == 0
    assert second["uploaded_files"] == 0
    assert second["skipped_existing_files"] == 4
    target = tmp_path / "pulled"
    pulled = pull_bundle(
        bundle["bundle_sha256"],
        target,
        client=client,
        bucket="bucket",
    )
    assert pulled["file_count"] == 4
    verify_checksums(target)
    assert (target / "metrics.json").read_bytes() == (source / "metrics.json").read_bytes()


def test_push_refuses_remote_collision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _run_bundle(source)
    bundle = build_run_bundle(source)
    client = _FakeS3()
    push_bundle(bundle, client=client, bucket="bucket")
    first_file = next(iter(bundle["files"]))
    key = f"chess-dfm/v1/bundles/sha256/{bundle['bundle_sha256']}/{first_file}"
    client.objects[("bucket", key)]["Metadata"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="collision"):
        push_bundle(bundle, client=client, bucket="bucket")


def test_dry_run_never_calls_client(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _run_bundle(source)
    bundle = build_run_bundle(source)
    result = push_bundle(bundle, client=None, bucket="bucket", dry_run=True)
    assert result["dry_run"] is True
    assert result["uploaded_files"] == 4
