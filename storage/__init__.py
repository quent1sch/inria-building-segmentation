"""
storage/__init__.py

Storage backend factory.
Returns the correct AbstractStorage implementation based on STORAGE_BACKEND env var.
"""

from config import get_settings
from storage.base import AbstractStorage


def get_storage() -> AbstractStorage:
    """
    Instantiate and return the configured storage backend.

    STORAGE_BACKEND=local  → LocalStorage (default, no cloud deps)
    STORAGE_BACKEND=azure  → AzureBlobStorage
    """
    settings = get_settings()

    if settings.storage_backend == "azure":
        if not settings.azure_storage_connection_string:
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING must be set when "
                "STORAGE_BACKEND=azure"
            )
        from storage.azure import AzureBlobStorage
        return AzureBlobStorage(
            connection_string=settings.azure_storage_connection_string,
            container_name=settings.azure_storage_container,
            url_expiry_s=settings.result_url_expiry_s,
        )

    # Default: local filesystem
    from storage.local import LocalStorage
    return LocalStorage(root=settings.storage_local_root)