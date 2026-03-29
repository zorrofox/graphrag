# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LiteLLM response cache backed by Google Cloud Storage.

Implements the graphrag Cache ABC. Also exposes the LiteLLM BaseCache methods
(set_cache, get_cache, async_set_cache, async_get_cache) so it can be plugged
directly into LiteLLM:

    from litellm import Cache
    import litellm
    cache = GCSLiteLLMCache(bucket_name="my-bucket", base_dir="llm-cache")
    litellm.cache = Cache(type="custom", cache_instance=cache)
"""

import json
import logging
from typing import TYPE_CHECKING, Any

from graphrag_cache.cache import Cache

if TYPE_CHECKING:
    from graphrag_storage import Storage

logger = logging.getLogger(__name__)


class GCSLiteLLMCache(Cache):
    """A graphrag Cache backed by Google Cloud Storage.

    Also exposes the LiteLLM BaseCache interface methods so it can be used
    as a custom LiteLLM cache backend.
    """

    def __init__(
        self,
        *,
        storage: "Storage | None" = None,
        bucket_name: str | None = None,
        base_dir: str = "litellm-cache",
        **kwargs: Any,
    ) -> None:
        from graphrag_storage.gcs_storage import GCSStorage

        if storage is not None:
            self._storage = storage
        elif bucket_name:
            self._storage: Storage = GCSStorage(
                bucket_name=bucket_name, base_dir=base_dir, **kwargs
            )
        else:
            msg = "GCSLiteLLMCache requires either a Storage instance or a bucket_name."
            raise ValueError(msg)

    def _key(self, key: str) -> str:
        """Normalize cache key to a safe GCS blob name."""
        return f"{key}.json"

    # ------------------------------------------------------------------
    # graphrag Cache ABC
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any:
        """Get a cached value by key."""
        return await self.async_get_cache(key)

    async def set(self, key: str, value: Any, debug_data: dict | None = None) -> None:
        """Set a cached value."""
        await self.async_set_cache(key, value)

    async def has(self, key: str) -> bool:
        """Return True if the key exists in the cache."""
        return await self._storage.has(self._key(key))

    async def delete(self, key: str) -> None:
        """Delete a cached key."""
        await self._storage.delete(self._key(key))

    async def clear(self) -> None:
        """Clear all cached entries."""
        await self._storage.clear()

    def child(self, name: str) -> "GCSLiteLLMCache":
        """Create a child cache with a nested GCS prefix."""
        return GCSLiteLLMCache(storage=self._storage.child(name))

    # ------------------------------------------------------------------
    # LiteLLM BaseCache interface (used when plugged into litellm.cache)
    # ------------------------------------------------------------------

    def set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        """Synchronous set — schedules async write or runs in event loop."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.async_set_cache(key, value, **kwargs))
            else:
                loop.run_until_complete(self.async_set_cache(key, value, **kwargs))
        except Exception:
            logger.warning(
                "GCSLiteLLMCache.set_cache failed for key %s", key, exc_info=True
            )

    def get_cache(self, key: str, **kwargs: Any) -> Any:
        """Synchronous get — returns None when called from a running event loop."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return None
            return loop.run_until_complete(self.async_get_cache(key, **kwargs))
        except Exception:
            logger.warning(
                "GCSLiteLLMCache.get_cache failed for key %s", key, exc_info=True
            )
            return None

    async def async_set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        """Store a JSON-serialised value in GCS."""
        try:
            await self._storage.set(self._key(key), json.dumps(value))
        except Exception:
            logger.warning(
                "GCSLiteLLMCache.async_set_cache failed for key %s", key, exc_info=True
            )

    async def async_get_cache(self, key: str, **kwargs: Any) -> Any:
        """Retrieve and deserialise a value from GCS; return None on miss."""
        try:
            raw = await self._storage.get(self._key(key))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            logger.warning(
                "GCSLiteLLMCache.async_get_cache failed for key %s", key, exc_info=True
            )
            return None

    async def async_set_cache_pipeline(
        self, cache_list: list[Any], **kwargs: Any
    ) -> None:
        """Store multiple key/value pairs (required by LiteLLM BaseCache ABC)."""
        for key, value in cache_list:
            await self.async_set_cache(key, value, **kwargs)

    async def batch_cache_write(self, key: str, value: Any, **kwargs: Any) -> None:
        """Delegate to async_set_cache."""
        await self.async_set_cache(key, value, **kwargs)

    async def disconnect(self) -> None:
        """Close the underlying GCS client."""
        self.close()

    def close(self) -> None:
        """Close the underlying GCS storage."""
        if hasattr(self._storage, "close"):
            self._storage.close()  # type: ignore[union-attr]
