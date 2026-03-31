# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Google Cloud Storage implementation of Storage."""

import asyncio
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from google.api_core import retry as api_retry
from google.api_core.exceptions import (
    InternalServerError,
    NotFound,
    ServiceUnavailable,
    TooManyRequests,
)
from google.cloud import storage

from graphrag_storage.storage import (
    Storage,
    get_timestamp_formatted_with_local_tz,
)

logger = logging.getLogger(__name__)

_GCS_RETRY: Any = api_retry.AsyncRetry(
    predicate=api_retry.if_exception_type(
        TooManyRequests, ServiceUnavailable, InternalServerError
    ),
    initial=1.0,
    maximum=60.0,
    multiplier=2.0,
    deadline=300.0,
)


class GCSStorage(Storage):
    """The Google Cloud Storage implementation."""

    _bucket_name: str
    _base_dir: str
    _encoding: str
    _client: storage.Client
    _bucket: storage.Bucket

    def __init__(self, **kwargs: Any) -> None:
        """Create a new GCSStorage instance."""
        bucket_name = kwargs.get("bucket_name")
        base_dir = kwargs.get("base_dir")
        credentials = kwargs.get("credentials")
        self._encoding = kwargs.get("encoding") or "utf-8"

        if not bucket_name:
            raise ValueError("No bucket_name provided for GCS storage.")

        self._bucket_name = bucket_name
        self._base_dir = base_dir or ""

        # Accept a pre-created client so child() can share it instead of
        # opening a new TCP connection for every sub-directory instance.
        _existing_client = kwargs.get("_client")
        if _existing_client is not None:
            self._client = _existing_client
        else:
            logger.info(
                "Creating GCS storage at bucket=%s, path=%s",
                self._bucket_name,
                self._base_dir,
            )
            self._client = storage.Client(credentials=credentials)

        self._bucket = self._client.bucket(self._bucket_name)

    def find(
        self,
        file_pattern: re.Pattern[str],
    ) -> Iterator[str]:
        """Find blobs in a bucket using a file pattern."""
        search_path = str(Path(self._base_dir) / "") if self._base_dir else ""
        if search_path == "./" or search_path == ".":
            search_path = ""
        if search_path and not search_path.endswith("/"):
            search_path += "/"

        logger.info(
            "Searching GCS bucket %s prefix %s for files matching %s",
            self._bucket_name,
            search_path,
            file_pattern.pattern,
        )

        try:
            blobs = self._client.list_blobs(
                self._bucket_name, prefix=search_path, page_size=1000
            )
            for blob in blobs:
                match = file_pattern.search(blob.name)
                if match:
                    yield self._relativize_path(blob.name)
        except Exception:
            logger.exception("Error finding blobs in GCS")
            raise

    async def get(
        self, key: str, as_bytes: bool | None = False, encoding: str | None = None
    ) -> Any:
        """Get a value from GCS."""
        try:
            blob = self._bucket.blob(self._keyname(key))
            data = await _GCS_RETRY(asyncio.to_thread)(blob.download_as_bytes)  # type: ignore[call-arg]
            if as_bytes:
                return data
            return data.decode(encoding or self._encoding)
        except NotFound:
            return None
        except Exception:
            logger.warning("Error getting key %s from GCS", key)
            return None

    async def set(self, key: str, value: Any, encoding: str | None = None) -> None:
        """Set a value in GCS."""
        try:
            blob = self._bucket.blob(self._keyname(key))
            if isinstance(value, str):
                coding = encoding or self._encoding
                await _GCS_RETRY(asyncio.to_thread)(
                    blob.upload_from_string,
                    value,
                    content_type=f"text/plain; charset={coding}",
                )
            elif isinstance(value, bytes):
                await _GCS_RETRY(asyncio.to_thread)(
                    blob.upload_from_string,
                    value,
                    content_type="application/octet-stream",
                )
            else:
                raise TypeError(f"Unsupported value type for GCS set: {type(value)}")
        except Exception:
            logger.exception("Error setting key %s in GCS", key)
            raise

    async def has(self, key: str) -> bool:
        """Check if a key exists in GCS."""
        try:
            blob = self._bucket.blob(self._keyname(key))
            return await _GCS_RETRY(asyncio.to_thread)(blob.exists)  # type: ignore[call-arg]
        except Exception:
            logger.warning("Error checking if key %s exists in GCS", key)
            return False

    async def delete(self, key: str) -> None:
        """Delete a key from GCS."""
        try:
            blob = self._bucket.blob(self._keyname(key))
            await _GCS_RETRY(asyncio.to_thread)(blob.delete)  # type: ignore[call-arg]
        except NotFound:
            pass
        except Exception:
            logger.exception("Error deleting key %s from GCS", key)
            raise

    async def clear(self) -> None:
        """Clear the storage (delete all blobs in base_dir)."""
        try:
            prefix = self._base_dir
            if prefix and not prefix.endswith("/"):
                prefix += "/"
            blobs = await _GCS_RETRY(asyncio.to_thread)(
                lambda: list(self._client.list_blobs(self._bucket_name, prefix=prefix))
            )
            if blobs:
                await _GCS_RETRY(asyncio.to_thread)(self._bucket.delete_blobs, blobs)  # type: ignore[call-arg]
        except Exception:
            logger.exception("Error clearing GCS storage at %s", self._base_dir)
            raise

    def child(self, name: str | None) -> "Storage":
        """Create a child storage instance sharing the same GCS client."""
        if name is None:
            return self
        new_base_dir = str(Path(self._base_dir) / name)
        return GCSStorage(
            bucket_name=self._bucket_name,
            base_dir=new_base_dir,
            encoding=self._encoding,
            _client=self._client,
        )

    def keys(self) -> list[str]:
        """List all keys in the storage."""
        try:
            prefix = self._base_dir
            if prefix and not prefix.endswith("/"):
                prefix += "/"
            blobs = self._client.list_blobs(
                self._bucket_name, prefix=prefix, page_size=1000
            )
            return [self._relativize_path(blob.name) for blob in blobs]
        except Exception:
            logger.exception("Error listing keys in GCS")
            raise

    async def get_creation_date(self, key: str) -> str:
        """Get the creation date for the given key."""
        try:
            blob = await _GCS_RETRY(asyncio.to_thread)(
                self._bucket.get_blob, self._keyname(key)
            )
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
        """Remove base_dir prefix from path."""
        prefix = self._base_dir
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        if path.startswith(prefix):
            return path[len(prefix):]
        return path

    def close(self) -> None:
        """Close the GCS client."""
        if self._client:
            self._client.close()
