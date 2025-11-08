# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import unittest
from unittest.mock import MagicMock

from graphrag.cache.json_pipeline_cache import JsonPipelineCache

class TestJsonPipelineCache(unittest.TestCase):
    def test_close(self):
        mock_storage = MagicMock()
        cache = JsonPipelineCache(mock_storage)
        
        cache.close()
        
        mock_storage.close.assert_called_once()

    def test_close_no_underlying_close(self):
        """Test that close() doesn't fail if underlying storage has no close method."""
        mock_storage = MagicMock()
        # MagicMock automatically creates methods when accessed, so we need to be careful.
        # A better way to simulate an object WITHOUT a close method is to use a plain object or a spec.
        class NoCloseStorage:
            pass
            
        cache = JsonPipelineCache(NoCloseStorage())
        # Should not raise AttributeError
        cache.close()
