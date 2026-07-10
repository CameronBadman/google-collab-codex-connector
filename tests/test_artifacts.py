from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from colab_codex_adapter.artifacts import (
    ArtifactNotFoundError,
    ArtifactStore,
)


def test_artifacts_are_private_and_read_in_bounded_chunks(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(root, max_artifact_bytes=1024, max_total_bytes=4096)
    ref = store.put_bytes(b"hello world", media_type="text/plain")

    assert root.stat().st_mode & 0o777 == 0o700
    assert (root / f"{ref.artifact_id}.artifact").stat().st_mode & 0o777 == 0o600
    assert (root / f"{ref.artifact_id}.json").stat().st_mode & 0o777 == 0o600
    first = store.read_chunk(ref.artifact_id, offset=0, limit_bytes=5)
    second = store.read_chunk(ref.artifact_id, offset=5, limit_bytes=64)
    assert first["data"] == "hello"
    assert first["next_offset"] == 5
    assert first["eof"] is False
    assert second["data"] == " world"
    assert second["eof"] is True


def test_binary_artifact_chunks_use_base64(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, max_artifact_bytes=1024, max_total_bytes=4096)
    ref = store.put_bytes(b"\xff\x00\xfe")

    chunk = store.read_chunk(ref.artifact_id, limit_bytes=3)

    assert chunk["encoding"] == "base64"
    assert chunk["data"] == "/wD+"


def test_artifact_size_cap_is_explicit(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, max_artifact_bytes=8, max_total_bytes=32)
    ref = store.put_bytes(b"abcdefghijkl")

    assert ref.size_bytes == 8
    assert ref.truncated is True
    assert store.read_chunk(ref.artifact_id, limit_bytes=8)["data"] == "abcdefgh"


def test_quota_evicts_oldest_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, max_artifact_bytes=8, max_total_bytes=12)
    first = store.put_bytes(b"12345678")
    time.sleep(0.01)
    second = store.put_bytes(b"abcdefgh")

    with pytest.raises(ArtifactNotFoundError):
        store.get_ref(first.artifact_id)
    assert store.get_ref(second.artifact_id) == second


def test_expired_and_malformed_ids_are_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(
        tmp_path, max_artifact_bytes=8, max_total_bytes=32, ttl_seconds=0.01
    )
    ref = store.put_bytes(b"value")
    time.sleep(0.02)

    with pytest.raises(ArtifactNotFoundError):
        store.read_chunk(ref.artifact_id)
    with pytest.raises(ArtifactNotFoundError):
        store.read_chunk("../../etc/passwd")
    assert not os.path.exists(tmp_path / ".." / "etc" / "passwd")


def test_unknown_artifact_errors_do_not_echo_supplied_id(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, max_artifact_bytes=8, max_total_bytes=32)
    supplied = "../../PRIVATE_ARTIFACT_ID"

    with pytest.raises(ArtifactNotFoundError) as error:
        store.read_chunk(supplied)

    assert str(error.value) == "Unknown or expired artifact id"
    assert supplied not in str(error.value)


def test_startup_cleanup_removes_orphans_and_temp_files(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    first = ArtifactStore(root, max_artifact_bytes=1024, max_total_bytes=4096)
    retained = first.put_bytes(b"retained")
    orphan_id = "a" * 32
    invalid_id = "b" * 32
    orphan_data = root / f"{orphan_id}.artifact"
    invalid_metadata = root / f"{invalid_id}.json"
    temporary = root / f".{orphan_id}.artifact.write.tmp"
    unrelated_json = root / "application.json"
    unrelated_artifact = root / "model.artifact"
    unrelated_temp = root / "application.tmp"
    orphan_data.write_bytes(b"orphan")
    invalid_metadata.write_text("not-json", encoding="utf-8")
    temporary.write_bytes(b"partial")
    unrelated_json.write_text("{}", encoding="utf-8")
    unrelated_artifact.write_bytes(b"model")
    unrelated_temp.write_bytes(b"temporary")

    restarted = ArtifactStore(
        root, max_artifact_bytes=1024, max_total_bytes=4096
    )

    assert restarted.get_ref(retained.artifact_id) == retained
    assert not orphan_data.exists()
    assert not invalid_metadata.exists()
    assert not temporary.exists()
    assert unrelated_json.exists()
    assert unrelated_artifact.exists()
    assert unrelated_temp.exists()


@pytest.mark.parametrize(
    "ttl_seconds", [float("nan"), float("inf"), 0, -1, True]
)
def test_artifact_ttl_must_be_finite_positive(
    tmp_path: Path, ttl_seconds: float
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        ArtifactStore(tmp_path, ttl_seconds=ttl_seconds)
