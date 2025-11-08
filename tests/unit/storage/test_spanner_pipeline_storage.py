# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import json
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from google.api_core import exceptions

from graphrag.storage.spanner_pipeline_storage import SpannerPipelineStorage


class TestSpannerPipelineStorage(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.project_id = "test-project"
        self.instance_id = "test-instance"
        self.database_id = "test-database"
        self.mock_client = MagicMock()
        self.mock_instance = MagicMock()
        self.mock_database = MagicMock()
        self.mock_client.instance.return_value = self.mock_instance
        self.mock_instance.database.return_value = self.mock_database

        with patch("google.cloud.spanner.Client", return_value=self.mock_client):
            self.storage = SpannerPipelineStorage(
                project_id=self.project_id,
                instance_id=self.instance_id,
                database_id=self.database_id,
            )

    def test_init_with_none_table_prefix(self):
        """Test initialization when table_prefix is explicitly None."""
        with patch("google.cloud.spanner.Client", return_value=self.mock_client):
            storage = SpannerPipelineStorage(
                project_id=self.project_id,
                instance_id=self.instance_id,
                database_id=self.database_id,
                table_prefix=None
            )
        self.assertEqual(storage._table_prefix, "")
        self.assertEqual(storage._blob_table, "Blobs")

    def test_init_missing_required_args(self):
        """Test that initialization fails if required arguments are missing."""
        with self.assertRaisesRegex(ValueError, "project_id, instance_id, and database_id are required"):
            SpannerPipelineStorage(project_id="p", instance_id="i")

    def test_init_empty_required_args(self):
        """Test that initialization fails if required arguments are empty strings."""
        with self.assertRaisesRegex(ValueError, "project_id, instance_id, and database_id are required"):
            SpannerPipelineStorage(project_id="", instance_id="i", database_id="d")

    async def test_set_table(self):
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        table_name = "TestTable"
        
        mock_batch = MagicMock()
        self.mock_database.batch.return_value.__enter__.return_value = mock_batch

        await self.storage.set_table(table_name, df)

        mock_batch.insert_or_update.assert_called_once()
        call_args = mock_batch.insert_or_update.call_args[1]
        self.assertEqual(call_args["table"], table_name)
        self.assertEqual(call_args["columns"], ("col1", "col2"))
        self.assertEqual(call_args["values"], [(1, "a"), (2, "b")])

    async def test_set_table_with_json(self):
        df = pd.DataFrame({
            "id": ["1"],
            "metadata": [{"key": "value"}],
            "tags": [["tag1", "tag2"]]
        })
        table_name = "JsonTable"
        
        mock_batch = MagicMock()
        self.mock_database.batch.return_value.__enter__.return_value = mock_batch

        # Mock schema to return JSON for metadata and ARRAY<STRING> for tags
        self.storage._schema_cache[table_name] = {
            "id": "STRING(MAX)",
            "metadata": "JSON",
            "tags": "ARRAY<STRING(MAX)>"
        }

        await self.storage.set_table(table_name, df)

        mock_batch.insert_or_update.assert_called_once()
        values = mock_batch.insert_or_update.call_args[1]["values"]
        self.assertEqual(values[0][0], "1")
        self.assertEqual(values[0][1], '{"key": "value"}') # Should be JSON string because schema says JSON
        self.assertEqual(values[0][2], ["tag1", "tag2"])   # Should remain list because schema says ARRAY

    async def test_set_table_json_retry(self):
        """Test that set_table retries with JSON dumping if it gets Expected JSON error."""
        df = pd.DataFrame({"id": ["1"], "json_col": [[]]}) # Empty list
        table_name = "JsonRetryTable"
        
        mock_batch = MagicMock()
        # First call raises FailedPrecondition (Expected JSON), second call succeeds
        mock_batch.insert_or_update.side_effect = [
            exceptions.FailedPrecondition("Invalid value for column json_col... Expected JSON"),
            None
        ]
        self.mock_database.batch.return_value.__enter__.return_value = mock_batch
        
        # Mock snapshot for schema query
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        
        # Use side_effect to return different schemas on subsequent calls
        schema_calls = 0
        def mock_execute_sql(*args, **kwargs):
            nonlocal schema_calls
            schema_calls += 1
            if schema_calls == 1:
                return iter([]) # Empty schema first time (simulating stale/unknown)
            else:
                return iter([["json_col", "JSON"]]) # Correct schema second time

        mock_results = MagicMock()
        mock_results.__iter__.side_effect = mock_execute_sql
        mock_snapshot.execute_sql.return_value = mock_results

        await self.storage.set_table(table_name, df)

        # Verify insert was retried
        self.assertEqual(mock_batch.insert_or_update.call_count, 2)
        
        # Verify second insert used JSON string for empty list
        call_args_2 = mock_batch.insert_or_update.call_args_list[1][1]
        # values is a list of tuples. [0] is first row, [1] is second column (json_col)
        self.assertEqual(call_args_2["values"][0][1], "[]")

    async def test_set_table_auto_create(self):
        """Test that set_table automatically creates the table if it doesn't exist."""
        df = pd.DataFrame({"id": ["1"], "col1": ["a"]})
        table_name = "AutoCreateTable"
        
        mock_batch = MagicMock()
        # First call raises NotFound, second call succeeds
        mock_batch.insert_or_update.side_effect = [
            exceptions.NotFound("Table not found"),
            None
        ]
        self.mock_database.batch.return_value.__enter__.return_value = mock_batch
        
        mock_operation = MagicMock()
        self.mock_database.update_ddl.return_value = mock_operation

        await self.storage.set_table(table_name, df)

        # Verify DDL was called
        self.mock_database.update_ddl.assert_called_once()
        ddl = self.mock_database.update_ddl.call_args[0][0][0]
        self.assertIn(f"CREATE TABLE `{table_name}`", ddl)
        self.assertIn("`id` STRING(MAX) NOT NULL", ddl)
        self.assertIn("`col1` STRING(MAX)", ddl)

        # Verify insert was retried
        self.assertEqual(mock_batch.insert_or_update.call_count, 2)

    async def test_set_table_auto_alter(self):
        """Test that set_table automatically alters the table if columns are missing."""
        df = pd.DataFrame({"id": ["1"], "col1": ["a"], "new_col": [100]})
        table_name = "AutoAlterTable"
        
        mock_batch = MagicMock()
        # First call raises NotFound (Column not found), second call succeeds
        mock_batch.insert_or_update.side_effect = [
            exceptions.NotFound("Column not found in table AutoAlterTable: new_col"),
            None
        ]
        self.mock_database.batch.return_value.__enter__.return_value = mock_batch
        
        # Mock snapshot for INFORMATION_SCHEMA query
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        
        # We need to handle multiple calls to execute_sql:
        # 1. _get_table_schema (initial) -> returns existing columns
        # 2. _alter_table_add_columns -> returns existing columns
        # 3. _get_table_schema (retry) -> returns ALL columns (including new one)
        
        call_count = 0
        def mock_execute_sql(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                 return iter([["id", "STRING(MAX)"], ["col1", "STRING(MAX)"]])
            else:
                 return iter([["id", "STRING(MAX)"], ["col1", "STRING(MAX)"], ["new_col", "INT64"]])

        mock_results = MagicMock()
        mock_results.__iter__.side_effect = mock_execute_sql
        mock_snapshot.execute_sql.return_value = mock_results

        mock_operation = MagicMock()
        self.mock_database.update_ddl.return_value = mock_operation

        await self.storage.set_table(table_name, df)

        # Verify ALTER TABLE DDL was called
        self.mock_database.update_ddl.assert_called_once()
        ddl = self.mock_database.update_ddl.call_args[0][0]
        self.assertIsInstance(ddl, list)
        self.assertEqual(len(ddl), 1)
        self.assertIn(f"ALTER TABLE `{table_name}` ADD COLUMN `new_col` INT64", ddl[0])

        # Verify insert was retried
        self.assertEqual(mock_batch.insert_or_update.call_count, 2)

    async def test_load_table(self):
        table_name = "TestTable"
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        
        # Mock StreamedResultSet
        mock_results = MagicMock()
        mock_results.__iter__.return_value = iter([[1, "a"], [2, "b"]])
        
        # Mock fields
        field1 = MagicMock()
        field1.name = "col1"
        field2 = MagicMock()
        field2.name = "col2"
        mock_results.fields = [field1, field2]
        
        mock_snapshot.execute_sql.return_value = mock_results

        df = await self.storage.load_table(table_name)

        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ["col1", "col2"])
        self.assertEqual(df.iloc[0]["col1"], 1)
        self.assertEqual(df.iloc[0]["col2"], "a")

    async def test_load_empty_table(self):
        table_name = "EmptyTable"
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        
        # Mock empty result but with fields
        mock_results = MagicMock()
        mock_results.__iter__.return_value = iter([])
        field1 = MagicMock()
        field1.name = "col1"
        mock_results.fields = [field1]
        
        mock_snapshot.execute_sql.return_value = mock_results

        df = await self.storage.load_table(table_name)

        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns), ["col1"])

    async def test_has_table_true(self):
        table_name = "TestTable"
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.execute_sql.return_value = [1] # Mock non-empty result

        self.assertTrue(await self.storage.has_table(table_name))

    async def test_has_table_false(self):
        table_name = "NonExistent"
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.execute_sql.side_effect = Exception("Table not found")

        self.assertFalse(await self.storage.has_table(table_name))

    async def test_blob_operations(self):
        key = "test.txt"
        value = b"hello"
        
        # Test set
        # Note: set() now uses run_in_transaction, not batch directly.
        # We need to mock run_in_transaction to call our callback.
        def mock_run_in_transaction(func, *args, **kwargs):
            mock_txn = MagicMock()
            func(mock_txn, *args, **kwargs)
            return None
        
        self.mock_database.run_in_transaction.side_effect = mock_run_in_transaction
        
        await self.storage.set(key, value)
        self.mock_database.run_in_transaction.assert_called()

        # Test get
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.read.return_value = [[value]]
        
        result = await self.storage.get(key, as_bytes=True)
        self.assertEqual(result, value)

        # Test has
        mock_snapshot.read.return_value = [["key"]]
        self.assertTrue(await self.storage.has(key))

        # Test delete
        mock_batch = MagicMock()
        self.mock_database.batch.return_value.__enter__.return_value = mock_batch
        await self.storage.delete(key)
        mock_batch.delete.assert_called()

    async def test_set_blob_auto_create(self):
        """Test that set() automatically creates the Blobs table if it doesn't exist."""
        key = "test.txt"
        value = b"hello"
        
        call_count = 0
        def mock_run_in_transaction(func, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise exceptions.NotFound("Table not found: Blobs")
            # Second call succeeds (mocked)
            return None

        self.mock_database.run_in_transaction.side_effect = mock_run_in_transaction
        
        mock_operation = MagicMock()
        self.mock_database.update_ddl.return_value = mock_operation

        await self.storage.set(key, value)

        # Verify DDL was called for Blobs table
        self.mock_database.update_ddl.assert_called_once()
        ddl = self.mock_database.update_ddl.call_args[0][0][0]
        self.assertIn("CREATE TABLE `Blobs`", ddl)

        # Verify transaction was retried
        self.assertEqual(call_count, 2)
