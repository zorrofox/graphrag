# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import unittest
from unittest.mock import MagicMock, patch
from graphrag.utils.spanner_client_manager import SpannerClientManager

class TestSpannerClientManager(unittest.TestCase):
    def setUp(self):
        # Reset the singleton state before each test
        SpannerClientManager._clients = {}
        SpannerClientManager._ref_counts = {}

    @patch("google.cloud.spanner.Client")
    def test_get_client_creates_new(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        client1 = SpannerClientManager.get_client(project_id="test-project")
        
        self.assertEqual(client1, mock_client)
        self.assertEqual(SpannerClientManager._ref_counts["test-project"], 1)
        mock_client_cls.assert_called_once_with(project="test-project", credentials=None)

    @patch("google.cloud.spanner.Client")
    def test_get_client_reuses_existing(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        client1 = SpannerClientManager.get_client(project_id="test-project")
        client2 = SpannerClientManager.get_client(project_id="test-project")
        
        self.assertEqual(client1, client2)
        self.assertEqual(SpannerClientManager._ref_counts["test-project"], 2)
        mock_client_cls.assert_called_once() # Only called once

    @patch("google.cloud.spanner.Client")
    def test_release_client_decrements_ref_count(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        client1 = SpannerClientManager.get_client(project_id="test-project")
        client2 = SpannerClientManager.get_client(project_id="test-project")
        
        SpannerClientManager.release_client(client1)
        self.assertEqual(SpannerClientManager._ref_counts["test-project"], 1)
        # Should NOT be closed yet
        mock_client.close.assert_not_called()

    @patch("google.cloud.spanner.Client")
    def test_release_client_closes_at_zero(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        client1 = SpannerClientManager.get_client(project_id="test-project")
        
        SpannerClientManager.release_client(client1)
        self.assertNotIn("test-project", SpannerClientManager._ref_counts)
        self.assertNotIn("test-project", SpannerClientManager._clients)
        # Should be closed now
        mock_client.close.assert_called_once()

    @patch("google.cloud.spanner.Client")
    def test_multiple_projects(self, mock_client_cls):
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        mock_client_cls.side_effect = [mock_client1, mock_client2]

        client1 = SpannerClientManager.get_client(project_id="project-1")
        client2 = SpannerClientManager.get_client(project_id="project-2")

        self.assertNotEqual(client1, client2)
        self.assertEqual(SpannerClientManager._ref_counts["project-1"], 1)
        self.assertEqual(SpannerClientManager._ref_counts["project-2"], 1)
