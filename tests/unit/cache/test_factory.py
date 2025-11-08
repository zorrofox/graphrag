# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import unittest
from unittest.mock import patch, MagicMock

from graphrag.cache.factory import CacheFactory
from graphrag.cache.json_pipeline_cache import JsonPipelineCache
from graphrag.config.enums import CacheType
from graphrag.storage.gcs_pipeline_storage import GCSPipelineStorage

class TestCacheFactory(unittest.TestCase):
    def test_create_gcs_cache(self):
        bucket_name = "test-bucket"
        base_dir = "cache"
        
        # Mock GCSPipelineStorage to avoid actual GCS connection attempts
        with patch("graphrag.storage.gcs_pipeline_storage.storage.Client"):
            cache = CacheFactory.create_cache(
                CacheType.gcs,
                {"bucket_name": bucket_name, "base_dir": base_dir}
            )
            
            self.assertIsInstance(cache, JsonPipelineCache)
            self.assertIsInstance(cache._storage, GCSPipelineStorage)
            self.assertEqual(cache._storage._bucket_name, bucket_name)
            self.assertEqual(cache._storage._base_dir, base_dir)

    def test_create_gcs_cache_missing_bucket(self):
        with self.assertRaises(ValueError) as cm:
             CacheFactory.create_cache(CacheType.gcs, {"base_dir": "cache"})
        self.assertIn("No bucket_name provided", str(cm.exception))
