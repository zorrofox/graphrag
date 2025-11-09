# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import unittest
from unittest.mock import MagicMock, patch
from graphrag.utils.spanner_resource_manager import SpannerResourceManager

class TestSpannerResourceManager(unittest.TestCase):
    def setUp(self):
        # Reset the singleton state before each test
        SpannerResourceManager._clients = {}
        SpannerResourceManager._databases = {}
        SpannerResourceManager._client_ref_counts = {}
        SpannerResourceManager._database_ref_counts = {}

    @patch("google.cloud.spanner.Client")
    def test_get_database_creates_new(self, mock_client_cls):
        mock_client = MagicMock()
        mock_instance = MagicMock()
        mock_database = MagicMock()
        
        mock_client_cls.return_value = mock_client
        mock_client.instance.return_value = mock_instance
        mock_instance.database.return_value = mock_database
        mock_database.name = "projects/p/instances/i/databases/d"

        db = SpannerResourceManager.get_database("p", "i", "d")
        
        self.assertEqual(db, mock_database)
        self.assertEqual(SpannerResourceManager._database_ref_counts[mock_database.name], 1)
        self.assertEqual(SpannerResourceManager._client_ref_counts["p"], 1)
        mock_client_cls.assert_called_once_with(project="p", credentials=None)

    @patch("google.cloud.spanner.Client")
    def test_get_database_reuses_existing(self, mock_client_cls):
        mock_client = MagicMock()
        mock_database = MagicMock()
        mock_database.name = "projects/p/instances/i/databases/d"
        
        mock_client_cls.return_value = mock_client
        mock_client.instance.return_value.database.return_value = mock_database

        db1 = SpannerResourceManager.get_database("p", "i", "d")
        db2 = SpannerResourceManager.get_database("p", "i", "d")
        
        self.assertEqual(db1, db2)
        self.assertEqual(SpannerResourceManager._database_ref_counts[mock_database.name], 2)
        self.assertEqual(SpannerResourceManager._client_ref_counts["p"], 1) # Client reused
        mock_client_cls.assert_called_once()

    @patch("google.cloud.spanner.Client")
    def test_release_database(self, mock_client_cls):
        mock_client = MagicMock()
        mock_database = MagicMock()
        mock_database.name = "projects/p/instances/i/databases/d"
        
        mock_client_cls.return_value = mock_client
        mock_client.instance.return_value.database.return_value = mock_database

        db = SpannerResourceManager.get_database("p", "i", "d")
        SpannerResourceManager.release_database(db)
        
        self.assertNotIn(mock_database.name, SpannerResourceManager._databases)
        self.assertNotIn("p", SpannerResourceManager._clients)
        mock_client.close.assert_called_once()

    @patch("google.cloud.spanner.Client")
    def test_release_database_keeps_client_if_shared(self, mock_client_cls):
        mock_client = MagicMock()
        mock_db1 = MagicMock()
        mock_db1.name = "projects/p/instances/i/databases/d1"
        mock_db2 = MagicMock()
        mock_db2.name = "projects/p/instances/i/databases/d2"
        
        mock_client_cls.return_value = mock_client
        # Return different DBs for different calls
        mock_client.instance.return_value.database.side_effect = [mock_db1, mock_db2]

        db1 = SpannerResourceManager.get_database("p", "i", "d1")
        db2 = SpannerResourceManager.get_database("p", "i", "d2")
        
        self.assertEqual(SpannerResourceManager._client_ref_counts["p"], 2)
        
        SpannerResourceManager.release_database(db1)
        self.assertNotIn(mock_db1.name, SpannerResourceManager._databases)
        self.assertEqual(SpannerResourceManager._client_ref_counts["p"], 1)
        mock_client.close.assert_not_called()
        
        SpannerResourceManager.release_database(db2)
        self.assertNotIn(mock_db2.name, SpannerResourceManager._databases)
        self.assertNotIn("p", SpannerResourceManager._clients)
        mock_client.close.assert_called_once()
