"""
storage/base.py

Abstract storage interface.

Both LocalStorage and AzureBlobStorage implement this interface.
The rest of the codebase only imports AbstractStorage and calls these
methods — swapping backends is a config change, not a code change.

get_url() is the key method that differs between backends:
  - LocalStorage:      returns a relative API URL (/results/{key})
                       the API streams the file from disk
  - AzureBlobStorage:  returns a time-limited presigned SAS URL
                       the client downloads directly from Azure
                       (the API never touches the bytes again)
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractStorage(ABC):

    @abstractmethod
    async def write(self, key: str, data: bytes) -> str:
        """
        Write bytes to storage under the given key.
        Returns the key (for confirmation).
        """

    @abstractmethod
    async def read(self, key: str) -> bytes:
        """Read bytes from storage by key."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Return True if key exists in storage."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a stored object."""

    @abstractmethod
    async def get_url(self, key: str) -> str:
        """
        Return a URL the client can use to download the result.

        Local:  /jobs/{key}/download  (served by the API)
        Azure:  https://....blob.core.windows.net/...?sas_token=...
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if storage is reachable. Used by /health endpoint."""