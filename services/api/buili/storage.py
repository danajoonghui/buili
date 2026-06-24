from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import UploadFile

from .config import get_settings


def object_path(r2_key: str) -> Path:
    settings = get_settings()
    safe_key = r2_key.strip("/").replace("..", "_")
    path = settings.storage_root / safe_key
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def save_upload(upload: UploadFile, r2_key: str, *, max_bytes: int) -> tuple[int, str]:
    path = object_path(r2_key)
    digest = hashlib.sha256()
    size = 0
    with path.open("wb") as fh:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                fh.close()
                path.unlink(missing_ok=True)
                raise ValueError("upload exceeds maximum allowed size")
            digest.update(chunk)
            fh.write(chunk)
    return size, digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
