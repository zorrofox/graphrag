# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Google Cloud Storage implementation of PipelineStorage."""

import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from google.cloud import storage
from google.cloud.storage.blob import Blob

from graphrag.storage.pipeline_storage import (
    PipelineStorage,
    get_timestamp_formatted_with_local_tz,
)

logger = logging.getLogger(__name__)


class GCSPipelineStorage(PipelineStorage):
    """The Google Cloud Storage implementation."""

    _bucket_name: str
    _base_dir: str
    _encoding: str
    _client: storage.Client
    _bucket: storage.Bucket

    def __init__(self, **kwargs: Any) -> None:
        """Create a new GCSPipelineStorage instance."""
        bucket_name = kwargs.get("bucket_name")
        base_dir = kwargs.get("base_dir")
        credentials = kwargs.get("credentials")
        self._encoding = kwargs.get("encoding", "utf-8")

        if not bucket_name:
            raise ValueError("No bucket_name provided for GCS storage.")

        self._bucket_name = bucket_name
        self._base_dir = base_dir or ""

        logger.info("Creating GCS storage at bucket=%s, path=%s", self._bucket_name, self._base_dir)

        self._client = storage.Client(credentials=credentials)
        self._bucket = self._client.bucket(self._bucket_name)

    def find(
        self,
        file_pattern: re.Pattern[str],
        base_dir: str | None = None,
        file_filter: dict[str, Any] | None = None,
        max_count: int = -1,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Find blobs in a bucket using a file pattern, as well as a custom filter function."""
        search_path = str(Path(self._base_dir) / (base_dir or ""))
        if search_path == ".":
            search_path = ""

        # Ensure search_path ends with / if it's not empty, to match directory-like behavior
        if search_path and not search_path.endswith("/"):
            search_path += "/"

        logger.info(
            "Searching GCS bucket %s prefix %s for files matching %s",
            self._bucket_name,
            search_path,
            file_pattern.pattern,
        )

        def _blobname(blob_name: str) -> str:
            # Remove the self._base_dir prefix to get the relative path
            if self._base_dir and blob_name.startswith(self._base_dir):
                # Handle potential trailing slash in base_dir
                prefix = self._base_dir
                if not prefix.endswith("/"):
                    prefix += "/"
                if blob_name.startswith(prefix):
                     return blob_name[len(prefix):]

            return blob_name

        def item_filter(item: dict[str, Any]) -> bool:
            if file_filter is None:
                return True
            return all(
                re.search(value, item[key]) for key, value in file_filter.items()
            )

        try:
            blobs = self._client.list_blobs(self._bucket_name, prefix=search_path)

            num_loaded = 0
            for blob in blobs:
                # We need to match against the full blob name or relative?
                # The base implementation seems to match against the full name but yields the relative name.
                # Let's stick to matching the full name for now as per BlobPipelineStorage
                match = file_pattern.search(blob.name)
                if match:
                    group = match.groupdict()
                    if item_filter(group):
                        yield (_blobname(blob.name), group)
                        num_loaded += 1
                        if max_count > 0 and num_loaded >= max_count:
                            break
        except Exception:
            logger.exception("Error finding blobs in GCS")
            raise

    async def get(
        self, key: str, as_bytes: bool | None = False, encoding: str | None = None
    ) -> Any:
        """Get a value from GCS."""
        try:
            blob_name = self._keyname(key)
            blob = self._bucket.blob(blob_name)
            
            if not blob.exists():
                return None

            data = blob.download_as_bytes()
            if as_bytes:
                return data
            
            coding = encoding or self._encoding
            return data.decode(coding)
        except Exception:
            logger.warning("Error getting key %s from GCS", key)
            return None

    async def set(self, key: str, value: Any, encoding: str | None = None) -> None:
        """Set a value in GCS."""
        try:
            blob_name = self._keyname(key)
            blob = self._bucket.blob(blob_name)
            
            if isinstance(value, str):
                coding = encoding or self._encoding
                blob.upload_from_string(value, content_type=f"text/plain; charset={coding}")
            elif isinstance(value, bytes):
                blob.upload_from_string(value, content_type="application/octet-stream")
            else:
                # Fallback for other types, try converting to string or bytes if possible,
                # or assume it's something acceptable by upload_from_string (like a file-like object? No, that's upload_from_file)
                # For now, assume it's str or bytes as per typical usage.
                raise TypeError(f"Unsupported value type for GCS set: {type(value)}")
                
        except Exception:
            logger.exception("Error setting key %s in GCS", key)
            raise

    async def has(self, key: str) -> bool:
        """Check if a key exists in GCS."""
        try:
            blob_name = self._keyname(key)
            blob = self._bucket.blob(blob_name)
            return blob.exists()
        except Exception:
            logger.warning("Error checking if key %s exists in GCS", key)
            return False

    async def delete(self, key: str) -> None:
        """Delete a key from GCS."""
        try:
            blob_name = self._keyname(key)
            blob = self._bucket.blob(blob_name)
            if blob.exists():
                blob.delete()
        except Exception:
             logger.exception("Error deleting key %s from GCS", key)
             raise

    async def clear(self) -> None:
        """Clear the storage (delete all blobs in base_dir)."""
        try:
            prefix = self._base_dir
            if prefix and not prefix.endswith("/"):
                prefix += "/"
            
            blobs = list(self._client.list_blobs(self._bucket_name, prefix=prefix))
            if blobs:
                self._bucket.delete_blobs(blobs)
        except Exception:
            logger.exception("Error clearing GCS storage at %s", self._base_dir)
            raise

    def child(self, name: str | None) -> "PipelineStorage":
        """Create a child storage instance."""
        if name is None:
            return self
        
        new_base_dir = str(Path(self._base_dir) / name)
        return GCSPipelineStorage(
            bucket_name=self._bucket_name,
            base_dir=new_base_dir,
            encoding=self._encoding,
            # Pass the same client/credentials if possible, but __init__ recreates client.
            # Optimization: could share client if passed explicitly.
            # For now, let it recreate to keep __init__ simple.
        )

    def keys(self) -> list[str]:
        """List all keys in the storage."""
        try:
            prefix = self._base_dir
            if prefix and not prefix.endswith("/"):
                prefix += "/"
            
            blobs = self._client.list_blobs(self._bucket_name, prefix=prefix)
            return [self._relativize_path(blob.name) for blob in blobs]
        except Exception:
            logger.exception("Error listing keys in GCS")
            raise

    async def get_creation_date(self, key: str) -> str:
        """Get the creation date for the given key."""
        try:
            blob_name = self._keyname(key)
            blob = self._bucket.get_blob(blob_name)
            if blob and blob.time_created:
                return get_timestamp_formatted_with_local_tz(blob.time_created)
            return ""
        except Exception:
            logger.warning("Error getting creation date for key %s in GCS", key)
            return ""

    def _keyname(self, key: str) -> str:
        """Get the full blob name including base_dir."""
        return str(Path(self._base_dir) / key)

    def _relativize_path(self, path: str) -> str:
        """Remove base_dir from path."""
        prefix = self._base_dir
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        
        if path.startswith(prefix):
            return path[len(prefix):]
        return path
