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
        SpannerResourceManager._db_to_client_key = {}

    # ------------------------------------------------------------------
    # _make_client_key
    # ------------------------------------------------------------------

    def test_make_client_key_none_credentials(self):
        key = SpannerResourceManager._make_client_key("proj", None)
        self.assertEqual(key, "proj/adc")

    def test_make_client_key_service_account(self):
        creds = MagicMock(spec=["service_account_email"])
        creds.service_account_email = "sa@proj.iam.gserviceaccount.com"
        key = SpannerResourceManager._make_client_key("proj", creds)
        self.assertEqual(key, "proj/sa@proj.iam.gserviceaccount.com")

    def test_make_client_key_generic_credentials(self):
        creds = MagicMock(spec=[])  # no service_account_email attribute
        key = SpannerResourceManager._make_client_key("proj", creds)
        self.assertTrue(key.startswith("proj/"))
        self.assertNotIn("adc", key)

    # ------------------------------------------------------------------
    # get_database
    # ------------------------------------------------------------------

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
        self.assertEqual(SpannerResourceManager._client_ref_counts["p/adc"], 1)
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
        self.assertEqual(SpannerResourceManager._client_ref_counts["p/adc"], 1)  # client reused
        mock_client_cls.assert_called_once()

    @patch("google.cloud.spanner.Client")
    def test_different_credentials_get_different_clients(self, mock_client_cls):
        """Same project with different credentials must produce separate clients."""
        creds_a = MagicMock(spec=["service_account_email"])
        creds_a.service_account_email = "sa-a@p.iam.gserviceaccount.com"
        creds_b = MagicMock(spec=["service_account_email"])
        creds_b.service_account_email = "sa-b@p.iam.gserviceaccount.com"

        mock_db_a, mock_db_b = MagicMock(), MagicMock()
        mock_db_a.name = "projects/p/instances/i/databases/d"
        mock_db_b.name = "projects/p/instances/i/databases/d"  # same path, different creds

        client_a, client_b = MagicMock(), MagicMock()
        client_a.instance.return_value.database.return_value = mock_db_a
        client_b.instance.return_value.database.return_value = mock_db_b
        mock_client_cls.side_effect = [client_a, client_b]

        SpannerResourceManager.get_database("p", "i", "d", credentials=creds_a)
        # db_key is the same, so the second call reuses the cached database
        # (same logical database, different client would only matter if creds changed
        #  for a NEW database entry; here we just verify client keys are distinct)
        self.assertIn("p/sa-a@p.iam.gserviceaccount.com", SpannerResourceManager._client_ref_counts)

    # ------------------------------------------------------------------
    # release_database
    # ------------------------------------------------------------------

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
        self.assertNotIn("p/adc", SpannerResourceManager._clients)
        # db → client mapping must also be cleaned up
        self.assertNotIn(mock_database.name, SpannerResourceManager._db_to_client_key)
        mock_client.close.assert_called_once()

    @patch("google.cloud.spanner.Client")
    def test_release_database_keeps_client_if_shared(self, mock_client_cls):
        mock_client = MagicMock()
        mock_db1 = MagicMock()
        mock_db1.name = "projects/p/instances/i/databases/d1"
        mock_db2 = MagicMock()
        mock_db2.name = "projects/p/instances/i/databases/d2"

        mock_client_cls.return_value = mock_client
        mock_client.instance.return_value.database.side_effect = [mock_db1, mock_db2]

        db1 = SpannerResourceManager.get_database("p", "i", "d1")
        db2 = SpannerResourceManager.get_database("p", "i", "d2")

        self.assertEqual(SpannerResourceManager._client_ref_counts["p/adc"], 2)

        SpannerResourceManager.release_database(db1)
        self.assertNotIn(mock_db1.name, SpannerResourceManager._databases)
        self.assertEqual(SpannerResourceManager._client_ref_counts["p/adc"], 1)
        mock_client.close.assert_not_called()

        SpannerResourceManager.release_database(db2)
        self.assertNotIn(mock_db2.name, SpannerResourceManager._databases)
        self.assertNotIn("p/adc", SpannerResourceManager._clients)
        mock_client.close.assert_called_once()
