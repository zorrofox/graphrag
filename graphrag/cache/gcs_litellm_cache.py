# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""LiteLLM response cache backed by Google Cloud Storage."""

import json
import logging
from typing import Any

from litellm.caching.caching import BaseCache

from graphrag.storage.gcs_pipeline_storage import GCSPipelineStorage

logger = logging.getLogger(__name__)


class GCSLiteLLMCache(BaseCache):
    """A LiteLLM BaseCache implementation that persists responses in GCS.

    Usage with LiteLLM:
        from litellm import Cache
        import litellm
        cache = GCSLiteLLMCache(bucket_name="my-bucket", base_dir="llm-cache")
        litellm.cache = Cache(type="custom", cache_instance=cache)
    """

    def __init__(
        self, bucket_name: str, base_dir: str = "litellm-cache", **kwargs: Any
    ) -> None:
        super().__init__()
        self._storage = GCSPipelineStorage(
            bucket_name=bucket_name, base_dir=base_dir, **kwargs
        )

    def _key(self, key: str) -> str:
        """Normalize cache key to a safe GCS blob name."""
        # LiteLLM keys can be long hashes — use as-is (GCS supports arbitrary names)
        return f"{key}.json"

    def set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        """Synchronous set — runs async in a new event loop."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule but don't await (fire-and-forget from sync context)
                loop.create_task(self.async_set_cache(key, value, **kwargs))
            else:
                loop.run_until_complete(self.async_set_cache(key, value, **kwargs))
        except Exception:
            logger.warning(
                "GCSLiteLLMCache.set_cache failed for key %s", key, exc_info=True
            )

    def get_cache(self, key: str, **kwargs: Any) -> Any:
        """Synchronous get — runs async in a new event loop."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return None  # Cannot block in running loop; return miss
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
                "GCSLiteLLMCache.async_get_cache failed for key %s",
                key,
                exc_info=True,
            )
            return None

    async def async_set_cache_pipeline(
        self, cache_list: list, **kwargs: Any
    ) -> None:
        """Store multiple key/value pairs; required by BaseCache ABC."""
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
        self._storage.close()
