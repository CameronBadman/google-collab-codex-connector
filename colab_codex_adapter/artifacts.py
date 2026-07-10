from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import secrets
import stat
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

DEFAULT_ARTIFACT_DIR = Path(
    os.environ.get(
        "COLAB_CODEX_ARTIFACT_DIR", "/tmp/colab-codex-adapter/artifacts"
    )
)
DEFAULT_MAX_ARTIFACT_BYTES = int(
    os.environ.get("COLAB_CODEX_MAX_ARTIFACT_BYTES", 32 * 1024 * 1024)
)
DEFAULT_MAX_ARTIFACT_TOTAL_BYTES = int(
    os.environ.get("COLAB_CODEX_MAX_ARTIFACT_TOTAL_BYTES", 256 * 1024 * 1024)
)
DEFAULT_ARTIFACT_TTL_SECONDS = float(
    os.environ.get("COLAB_CODEX_ARTIFACT_TTL_SECONDS", 24 * 60 * 60)
)
DEFAULT_ARTIFACT_READ_BYTES = 64 * 1024
MAX_ARTIFACT_READ_BYTES = 64 * 1024
UNKNOWN_ARTIFACT_MESSAGE = "Unknown or expired artifact id"
_ATOMIC_TEMP_PATTERN = re.compile(
    r"^\.[0-9a-f]{32}\.(?:artifact|json)\..+\.tmp$"
)


def valid_opaque_id(value: object) -> bool:
    """Return whether value is one of the connector's lowercase opaque IDs."""

    return isinstance(value, str) and len(value) == 32 and all(
        char in "0123456789abcdef" for char in value
    )


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    storage: str
    media_type: str
    size_bytes: int
    sha256: str
    created_at: float
    expires_at: float | None
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ArtifactRecord:
    ref: ArtifactRef
    data_path: Path
    metadata_path: Path


class ArtifactNotFoundError(ValueError):
    pass


class ArtifactStore:
    """Private, quota-bound storage addressed only by issued opaque IDs."""

    def __init__(
        self,
        root: Path | str = DEFAULT_ARTIFACT_DIR,
        *,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        max_total_bytes: int = DEFAULT_MAX_ARTIFACT_TOTAL_BYTES,
        ttl_seconds: float = DEFAULT_ARTIFACT_TTL_SECONDS,
    ) -> None:
        if (
            isinstance(max_artifact_bytes, bool)
            or not isinstance(max_artifact_bytes, int)
            or max_artifact_bytes <= 0
        ):
            raise ValueError("max_artifact_bytes must be greater than zero")
        if (
            isinstance(max_total_bytes, bool)
            or not isinstance(max_total_bytes, int)
            or max_total_bytes < max_artifact_bytes
        ):
            raise ValueError(
                "max_total_bytes must be greater than or equal to "
                "max_artifact_bytes"
            )
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int | float)
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be a finite positive number")
        self.root = Path(root)
        self.max_artifact_bytes = max_artifact_bytes
        self.max_total_bytes = max_total_bytes
        self.ttl_seconds = ttl_seconds
        self._lock = Lock()
        self._ensure_root()
        self.cleanup()

    def _ensure_root(self) -> None:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def _data_path(self, artifact_id: str) -> Path:
        return self.root / f"{artifact_id}.artifact"

    def _metadata_path(self, artifact_id: str) -> Path:
        return self.root / f"{artifact_id}.json"

    @staticmethod
    def _valid_id(artifact_id: object) -> bool:
        return valid_opaque_id(artifact_id)

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            os.chmod(path, 0o600)
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            temporary_path.unlink(missing_ok=True)
            raise

    def _read_record(self, metadata_path: Path) -> _ArtifactRecord | None:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            ref = ArtifactRef(**metadata)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if (
            not self._valid_id(ref.artifact_id)
            or metadata_path.name != f"{ref.artifact_id}.json"
            or not isinstance(ref.storage, str)
            or not isinstance(ref.media_type, str)
            or isinstance(ref.size_bytes, bool)
            or not isinstance(ref.size_bytes, int)
            or ref.size_bytes < 0
            or not _valid_sha256(ref.sha256)
            or isinstance(ref.created_at, bool)
            or not isinstance(ref.created_at, int | float)
            or (
                ref.expires_at is not None
                and (
                    isinstance(ref.expires_at, bool)
                    or not isinstance(ref.expires_at, int | float)
                )
            )
            or not isinstance(ref.truncated, bool)
        ):
            return None
        data_path = self._data_path(ref.artifact_id)
        try:
            data_stat = data_path.lstat()
        except OSError:
            return None
        if not stat.S_ISREG(data_stat.st_mode) or data_stat.st_size != ref.size_bytes:
            return None
        return _ArtifactRecord(ref, data_path, metadata_path)

    @staticmethod
    def _delete_record(record: _ArtifactRecord) -> None:
        record.data_path.unlink(missing_ok=True)
        record.metadata_path.unlink(missing_ok=True)

    def _cleanup_locked(self, required_bytes: int = 0) -> None:
        now = time.time()
        retained: list[_ArtifactRecord] = []
        retained_ids: set[str] = set()

        for metadata_path in self.root.glob("*.json"):
            if not valid_opaque_id(metadata_path.stem):
                continue
            record = self._read_record(metadata_path)
            if record is None:
                metadata_path.unlink(missing_ok=True)
                continue
            expires_at = record.ref.expires_at
            if expires_at is not None and expires_at <= now:
                self._delete_record(record)
            else:
                retained.append(record)
                retained_ids.add(record.ref.artifact_id)

        for data_path in self.root.glob("*.artifact"):
            if valid_opaque_id(data_path.stem) and data_path.stem not in retained_ids:
                data_path.unlink(missing_ok=True)

        for path in self.root.iterdir():
            if (
                _ATOMIC_TEMP_PATTERN.fullmatch(path.name)
                and not path.is_dir()
            ):
                path.unlink(missing_ok=True)

        total = sum(record.ref.size_bytes for record in retained)
        for record in sorted(retained, key=lambda item: item.ref.created_at):
            if total + required_bytes <= self.max_total_bytes:
                break
            self._delete_record(record)
            total -= record.ref.size_bytes

    def cleanup(self) -> None:
        with self._lock:
            self._cleanup_locked()

    def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
    ) -> ArtifactRef:
        self._ensure_root()
        original_size = len(data)
        stored = data[: self.max_artifact_bytes]
        created_at = time.time()
        artifact_id = secrets.token_hex(16)
        ref = ArtifactRef(
            artifact_id=artifact_id,
            storage="broker",
            media_type=media_type,
            size_bytes=len(stored),
            sha256=hashlib.sha256(stored).hexdigest(),
            created_at=created_at,
            expires_at=created_at + self.ttl_seconds,
            truncated=original_size > len(stored),
        )
        metadata = json.dumps(
            ref.to_dict(), separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        with self._lock:
            self._cleanup_locked(required_bytes=len(stored))
            self._write_atomic(self._data_path(artifact_id), stored)
            try:
                self._write_atomic(self._metadata_path(artifact_id), metadata)
            except BaseException:
                self._data_path(artifact_id).unlink(missing_ok=True)
                raise
        return ref

    def put_json(self, value: Any) -> ArtifactRef:
        return self.put_bytes(
            json_bytes(value), media_type="application/json; charset=utf-8"
        )

    def get_ref(self, artifact_id: str) -> ArtifactRef:
        if not self._valid_id(artifact_id):
            raise ArtifactNotFoundError(UNKNOWN_ARTIFACT_MESSAGE)
        with self._lock:
            self._cleanup_locked()
            record = self._read_record(self._metadata_path(artifact_id))
        if record is None:
            raise ArtifactNotFoundError(UNKNOWN_ARTIFACT_MESSAGE)
        return record.ref

    def read_chunk(
        self,
        artifact_id: str,
        *,
        offset: int = 0,
        limit_bytes: int = DEFAULT_ARTIFACT_READ_BYTES,
    ) -> dict[str, Any]:
        if not self._valid_id(artifact_id):
            raise ArtifactNotFoundError(UNKNOWN_ARTIFACT_MESSAGE)
        if offset < 0:
            raise ValueError("offset must be greater than or equal to zero")
        if not 1 <= limit_bytes <= MAX_ARTIFACT_READ_BYTES:
            raise ValueError(
                f"limit_bytes must be between 1 and {MAX_ARTIFACT_READ_BYTES}"
            )
        ref = self.get_ref(artifact_id)
        if offset > ref.size_bytes:
            raise ValueError("offset is beyond the end of the artifact")
        path = self._data_path(artifact_id)
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read(limit_bytes)
        except OSError as exc:
            raise ArtifactNotFoundError(UNKNOWN_ARTIFACT_MESSAGE) from exc

        next_offset = offset + len(chunk)
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            encoding = "base64"
            data = base64.b64encode(chunk).decode("ascii")
        else:
            encoding = "utf-8"
            data = text
        return {
            "artifact": ref.to_dict(),
            "offset": offset,
            "next_offset": next_offset,
            "eof": next_offset >= ref.size_bytes,
            "encoding": encoding,
            "data": data,
        }


def _json_default(value: Any) -> Any:
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True)
    return repr(value)


def json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


_default_store: ArtifactStore | None = None
_default_store_lock = Lock()


def get_default_artifact_store() -> ArtifactStore:
    global _default_store
    with _default_store_lock:
        if _default_store is None:
            _default_store = ArtifactStore()
        return _default_store
