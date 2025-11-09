# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import json
import unittest
from unittest.mock import MagicMock, patch, call

from google.cloud.spanner_v1 import param_types
from google.api_core import exceptions

from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.vector_stores.base import VectorStoreDocument
from graphrag.vector_stores.spanner import SpannerVectorStore


class TestSpannerVectorStore(unittest.TestCase):
    def setUp(self):
        self.project_id = "test-project"
        self.instance_id = "test-instance"
        self.database_id = "test-database"
        self.index_name = "test_table"
        self.config = VectorStoreSchemaConfig(
            index_name=self.index_name,
            id_field="id",
            text_field="text",
            vector_field="vector",
            attributes_field="attributes",
            vector_size=3,
        )
        self.mock_client = MagicMock()
        self.mock_instance = MagicMock()
        self.mock_database = MagicMock()
        self.mock_client.instance.return_value = self.mock_instance
        self.mock_instance.database.return_value = self.mock_database

        # Use patch for SpannerClientManager.get_client
        self.client_manager_patcher = patch("graphrag.vector_stores.spanner.SpannerClientManager")
        self.mock_client_manager = self.client_manager_patcher.start()
        self.mock_client_manager.get_client.return_value = self.mock_client

        self.store = SpannerVectorStore(
            vector_store_schema_config=self.config,
            project_id=self.project_id,
            instance_id=self.instance_id,
            database_id=self.database_id,
        )
        self.store.connect()

    def tearDown(self):
        self.client_manager_patcher.stop()

    def test_connect(self):
        self.mock_client.instance.assert_called_with(self.instance_id)
        self.mock_instance.database.assert_called_with(self.database_id)

    def test_close(self):
        """Test that close() calls SpannerClientManager.release_client."""
        self.store.close()
        # Should NOT call client.close() directly anymore
        self.mock_client.close.assert_not_called()
        # Should call release_client instead
        self.mock_client_manager.release_client.assert_called_once_with(self.mock_client)

    def test_load_documents(self):
        docs = [
            VectorStoreDocument(
                id="1", text="doc1", vector=[0.1, 0.2, 0.3], attributes={"key": "value"}
            ),
            VectorStoreDocument(id="2", text="doc2", vector=[0.4, 0.5, 0.6]),
        ]
        mock_batch = MagicMock()
        self.mock_database.batch.return_value.__enter__.return_value = mock_batch

        self.store.load_documents(docs)

        expected_rows = [
            ("1", "doc1", [0.1, 0.2, 0.3], '{"key": "value"}'),
            ("2", "doc2", [0.4, 0.5, 0.6], None),
        ]
        mock_batch.insert_or_update.assert_called_with(
            table=self.index_name,
            columns=("id", "text", "vector", "attributes"),
            values=expected_rows,
        )

    def test_filter_by_id(self):
        ids = ["1", "2"]
        self.store.filter_by_id(ids)
        self.assertEqual(self.store.query_filter, ids)

    def test_similarity_search_by_vector(self):
        query_vector = [0.1, 0.1, 0.1]
        k = 5
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        
        # Mock results: id, text, vector, attributes, distance
        # Spanner client returns dicts for JSON columns, not strings
        mock_snapshot.execute_sql.return_value = [
            ["1", "doc1", [0.1, 0.2, 0.3], {"key": "value"}, 0.1],
            ["2", "doc2", [0.4, 0.5, 0.6], None, 0.5],
        ]

        results = self.store.similarity_search_by_vector(query_vector, k=k)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].document.id, "1")
        self.assertEqual(results[0].score, 0.9)  # 1 - 0.1
        self.assertEqual(results[0].document.attributes, {"key": "value"})
        self.assertEqual(results[1].document.id, "2")
        self.assertEqual(results[1].score, 0.5)  # 1 - 0.5
        self.assertEqual(results[1].document.attributes, {})

        # Verify SQL call without filter
        args, kwargs = mock_snapshot.execute_sql.call_args
        sql = args[0]
        self.assertIn(f"FROM {self.index_name}", sql)
        self.assertIn("ORDER BY distance", sql)
        self.assertNotIn("WHERE id IN UNNEST", sql)
        self.assertEqual(kwargs["params"]["k"], k)
        self.assertEqual(kwargs["params"]["query_vector"], query_vector)

    def test_similarity_search_by_vector_with_filter(self):
        query_vector = [0.1, 0.1, 0.1]
        k = 5
        filter_ids = ["1"]
        self.store.filter_by_id(filter_ids)
        
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.execute_sql.return_value = []

        self.store.similarity_search_by_vector(query_vector, k=k)

        # Verify SQL call with filter
        args, kwargs = mock_snapshot.execute_sql.call_args
        sql = args[0]
        self.assertIn("WHERE id IN UNNEST(@include_ids)", sql)
        self.assertEqual(kwargs["params"]["include_ids"], filter_ids)

    def test_search_by_id(self):
        doc_id = "1"
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.execute_sql.return_value = [
            ["1", "doc1", [0.1, 0.2, 0.3], {"key": "value"}]
        ]

        doc = self.store.search_by_id(doc_id)

        self.assertEqual(doc.id, "1")
        self.assertEqual(doc.text, "doc1")
        self.assertEqual(doc.attributes, {"key": "value"})

        args, kwargs = mock_snapshot.execute_sql.call_args
        sql = args[0]
        self.assertIn("WHERE id = @id", sql)
        self.assertEqual(kwargs["params"]["id"], doc_id)

    def test_search_by_id_not_found(self):
        doc_id = "non_existent"
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.execute_sql.return_value = []

        doc = self.store.search_by_id(doc_id)

        self.assertEqual(doc.id, doc_id)
        self.assertIsNone(doc.text)
        self.assertIsNone(doc.vector)

    def test_load_documents_auto_create_table(self):
        docs = [VectorStoreDocument(id="1", text="doc1", vector=[0.1, 0.2, 0.3])]
        
        mock_batch = MagicMock()
        self.mock_database.batch.return_value.__enter__.return_value = mock_batch
        
        # First call raises NotFound, second call succeeds
        mock_batch.insert_or_update.side_effect = [
            exceptions.NotFound("Table not found"),
            None
        ]

        mock_operation = MagicMock()
        self.mock_database.update_ddl.return_value = mock_operation

        self.store.load_documents(docs)

        # Verify DDL was called with BOTH table and index creation statements
        self.mock_database.update_ddl.assert_called_once()
        ddl_list = self.mock_database.update_ddl.call_args[0][0]
        self.assertEqual(len(ddl_list), 2)
        
        # Check table DDL
        self.assertIn(f"CREATE TABLE IF NOT EXISTS `{self.index_name}`", ddl_list[0])
        self.assertIn(f"ARRAY<FLOAT64>(vector_length=>{self.config.vector_size})", ddl_list[0])
        
        # Check index DDL
        self.assertIn(f"CREATE VECTOR INDEX IF NOT EXISTS `{self.index_name}_VectorIndex`", ddl_list[1])
        self.assertIn(f"WHERE `{self.config.vector_field}` IS NOT NULL", ddl_list[1])
        self.assertIn("OPTIONS (distance_type = 'COSINE')", ddl_list[1])

        # Verify insert_or_update was called twice
        self.assertEqual(mock_batch.insert_or_update.call_count, 2)

    def test_init_sanitizes_table_name(self):
        # Test that hyphens in index_name are replaced with underscores
        config = VectorStoreSchemaConfig(
            index_name="table-with-hyphens",
            id_field="id",
            text_field="text",
            vector_field="vector",
            attributes_field="attributes",
            vector_size=3,
        )
        # We need to patch SpannerClientManager here too because it's used in __init__
        with patch("graphrag.vector_stores.spanner.SpannerClientManager"):
            store = SpannerVectorStore(
                vector_store_schema_config=config,
                project_id=self.project_id,
                instance_id=self.instance_id,
                database_id=self.database_id,
            )
            self.assertEqual(store.index_name, "table_with_hyphens")