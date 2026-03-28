# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for GCSLiteLLMCache."""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class TestGCSLiteLLMCache(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with patch(
            "graphrag.cache.gcs_litellm_cache.GCSPipelineStorage"
        ) as mock_storage_cls:
            self.mock_storage = MagicMock()
            mock_storage_cls.return_value = self.mock_storage
            from graphrag.cache.gcs_litellm_cache import GCSLiteLLMCache

            self.cache = GCSLiteLLMCache(bucket_name="test-bucket", base_dir="test")

    async def test_async_set_cache_serializes_to_json(self):
        self.mock_storage.set = AsyncMock()
        value = {"role": "assistant", "content": "hello"}
        await self.cache.async_set_cache("key123", value)
        self.mock_storage.set.assert_called_once_with("key123.json", json.dumps(value))

    async def test_async_get_cache_deserializes_json(self):
        value = {"role": "assistant", "content": "hello"}
        self.mock_storage.get = AsyncMock(return_value=json.dumps(value))
        result = await self.cache.async_get_cache("key123")
        self.assertEqual(result, value)

    async def test_async_get_cache_returns_none_on_miss(self):
        self.mock_storage.get = AsyncMock(return_value=None)
        result = await self.cache.async_get_cache("missing")
        self.assertIsNone(result)

    async def test_async_get_cache_returns_none_on_error(self):
        self.mock_storage.get = AsyncMock(side_effect=Exception("GCS error"))
        result = await self.cache.async_get_cache("key")
        self.assertIsNone(result)

    async def test_async_set_cache_swallows_errors(self):
        self.mock_storage.set = AsyncMock(side_effect=Exception("GCS error"))
        # Should not raise
        await self.cache.async_set_cache("key", {"data": "value"})

    def test_key_adds_json_extension(self):
        self.assertEqual(self.cache._key("abc123"), "abc123.json")

    async def test_async_set_cache_pipeline_calls_async_set_cache(self):
        self.mock_storage.set = AsyncMock()
        pairs = [("k1", {"v": 1}), ("k2", {"v": 2})]
        await self.cache.async_set_cache_pipeline(pairs)
        self.assertEqual(self.mock_storage.set.call_count, 2)
        self.mock_storage.set.assert_any_call("k1.json", json.dumps({"v": 1}))
        self.mock_storage.set.assert_any_call("k2.json", json.dumps({"v": 2}))

    async def test_batch_cache_write_delegates(self):
        self.mock_storage.set = AsyncMock()
        value = {"x": 42}
        await self.cache.batch_cache_write("bkey", value)
        self.mock_storage.set.assert_called_once_with("bkey.json", json.dumps(value))

    async def test_disconnect_calls_close(self):
        self.mock_storage.close = MagicMock()
        await self.cache.disconnect()
        self.mock_storage.close.assert_called_once()

    def test_close_delegates_to_storage(self):
        self.mock_storage.close = MagicMock()
        self.cache.close()
        self.mock_storage.close.assert_called_once()
