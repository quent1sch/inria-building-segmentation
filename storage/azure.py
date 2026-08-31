"""
storage/azure.py

Azure Blob Storage backend.

Uses the azure-storage-blob SDK (async client).
get_url() returns a time-limited SAS URL — the client downloads
the result directly from Azure without the API touching the bytes.
This is the cloud-native pattern: API is stateless, storage is
decoupled, and large files never flow back through the API server.

Required env vars:
    AZURE_STORAGE_CONNECTION_STRING   from Azure Portal → Storage Account → Access keys
    AZURE_STORAGE_CONTAINER           blob container name (created if not exists)

Optional:
    RESULT_URL_EXPIRY_S               SAS URL validity in seconds (default 3600)

Install:
    pip install azure-storage-blob
"""

from __future__ import annotations

from datetime import timedelta, timezone, datetime

from storage.base import AbstractStorage


class AzureBlobStorage(AbstractStorage):

    def __init__(
        self,
        connection_string: str,
        container_name: str,
        url_expiry_s: int = 3600,
    ):
        # Import here so the package is optional — only needed when
        # STORAGE_BACKEND=azure. Local mode works without it.
        try:
            from azure.storage.blob.aio import BlobServiceClient
            from azure.storage.blob import generate_blob_sas, BlobSasPermissions
        except ImportError:
            raise ImportError(
                "azure-storage-blob is required for Azure storage. "
                "pip install azure-storage-blob"
            )

        self._conn_str      = connection_string
        self._container     = container_name
        self._url_expiry_s  = url_expiry_s
        self._BlobServiceClient   = BlobServiceClient
        self._generate_blob_sas   = generate_blob_sas
        self._BlobSasPermissions  = BlobSasPermissions

    def _client(self):
        return self._BlobServiceClient.from_connection_string(self._conn_str)

    async def _ensure_container(self, client) -> None:
        container = client.get_container_client(self._container)
        try:
            await container.get_container_properties()
        except Exception:
            await container.create_container()

    async def write(self, key: str, data: bytes) -> str:
        async with self._client() as client:
            await self._ensure_container(client)
            blob = client.get_blob_client(container=self._container, blob=key)
            await blob.upload_blob(data, overwrite=True)
        return key

    async def read(self, key: str) -> bytes:
        async with self._client() as client:
            blob = client.get_blob_client(container=self._container, blob=key)
            stream = await blob.download_blob()
            return await stream.readall()

    async def exists(self, key: str) -> bool:
        async with self._client() as client:
            blob = client.get_blob_client(container=self._container, blob=key)
            return await blob.exists()

    async def delete(self, key: str) -> None:
        async with self._client() as client:
            blob = client.get_blob_client(container=self._container, blob=key)
            await blob.delete_blob(delete_snapshots="include")

    async def get_url(self, key: str) -> str:
        """
        Generate a time-limited SAS URL.
        The client downloads directly from Azure — the API never touches
        the result bytes again. This is the correct cloud-native pattern.

        SAS URL format:
        https://<account>.blob.core.windows.net/<container>/<key>?<sas_token>
        """
        from azure.storage.blob import (
            BlobSasPermissions,
            generate_blob_sas,
            BlobServiceClient as SyncClient,
        )

        # Parse account name and key from connection string for SAS generation
        # (SAS generation is synchronous in the SDK)
        sync_client  = SyncClient.from_connection_string(self._conn_str)
        account_name = sync_client.account_name
        account_key  = sync_client.credential.account_key

        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=self._container,
            blob_name=key,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(seconds=self._url_expiry_s),
        )
        return (
            f"https://{account_name}.blob.core.windows.net"
            f"/{self._container}/{key}?{sas_token}"
        )

    async def health_check(self) -> bool:
        try:
            async with self._client() as client:
                await self._ensure_container(client)
                return True
        except Exception:
            return False