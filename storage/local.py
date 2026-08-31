"""
storage/local.py

Local filesystem storage backend.

Stores result files under STORAGE_LOCAL_ROOT/{key}.
get_url() returns an internal API URL — the API streams the file
from disk when the client fetches it.

Used for local development and single-server deployments.
No external dependencies.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiofiles

from storage.base import AbstractStorage


class LocalStorage(AbstractStorage):

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Sanitize key — prevent path traversal
        safe_key = Path(key).name
        return self.root / safe_key

    async def write(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        return key

    async def read(self, key: str) -> bytes:
        async with aiofiles.open(self._path(key), "rb") as f:
            return await f.read()

    async def exists(self, key: str) -> bool:
        return self._path(key).exists()

    async def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()

    async def get_url(self, key: str) -> str:
        """
        Returns an internal API URL.
        The /jobs/{job_id}/result endpoint streams the file from disk.
        In production with Azure this would be a presigned blob URL instead.
        """
        return f"/results/{key}"

    async def health_check(self) -> bool:
        try:
            test_key = "_health_check"
            await self.write(test_key, b"ok")
            data = await self.read(test_key)
            await self.delete(test_key)
            return data == b"ok"
        except Exception:
            return False