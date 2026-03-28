# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import base64
import json
import re
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
from google.api_core import exceptions

from graphrag.storage.spanner_pipeline_storage import (
    SpannerPipelineStorage,
    _safe_identifier,
)


class TestSpannerPipelineStorage(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.project_id = "test-project"
        self.instance_id = "test-instance"
        self.database_id = "test-database"
        self.mock_database = MagicMock()

        # Use patch for SpannerResourceManager.get_database
        self.resource_manager_patcher = patch("graphrag.storage.spanner_pipeline_storage.SpannerResourceManager")
        self.mock_resource_manager = self.resource_manager_patcher.start()
        self.mock_resource_manager.get_database.return_value = self.mock_database

        self.storage = SpannerPipelineStorage(
            project_id=self.project_id,
            instance_id=self.instance_id,
            database_id=self.database_id,
        )

    def tearDown(self):
        self.resource_manager_patcher.stop()

    def test_init_with_none_table_prefix(self):
        """Test initialization when table_prefix is explicitly None."""
        storage = SpannerPipelineStorage(
            project_id=self.project_id,
            instance_id=self.instance_id,
            database_id=self.database_id,
            table_prefix=None
        )
        self.assertEqual(storage._table_prefix, "")
        self.assertEqual(storage._blob_table, "Blobs")

    def test_init_sanitizes_table_prefix(self):
        """Test that table_prefix is sanitized during initialization."""
        storage = SpannerPipelineStorage(
            project_id=self.project_id,
            instance_id=self.instance_id,
            database_id=self.database_id,
            table_prefix="my-app."
        )
        self.assertEqual(storage._table_prefix, "my_app_")
        self.assertEqual(storage._blob_table, "my_app_Blobs")

    def test_init_missing_required_args(self):
        """Test that initialization fails if required arguments are missing."""
        with self.assertRaisesRegex(ValueError, "project_id, instance_id, and database_id are required"):
            SpannerPipelineStorage(project_id="p", instance_id="i")

    def test_close(self):
        """Test that close() calls SpannerResourceManager.release_database."""
        self.storage.close()
        self.mock_resource_manager.release_database.assert_called_once_with(self.mock_database)
        self.assertIsNone(self.storage._database)

    def test_init_empty_required_args(self):
        """Test that initialization fails if required arguments are empty strings."""
        with self.assertRaisesRegex(ValueError, "project_id, instance_id, and database_id are required"):
            SpannerPipelineStorage(project_id="", instance_id="i", database_id="d")

    def test_sanitize_table_name(self):
        """Test that table names are correctly sanitized."""
        self.assertEqual(self.storage._sanitize_table_name("normal_table"), "normal_table")
        self.assertEqual(self.storage._sanitize_table_name("table-with-hyphens"), "table_with_hyphens")
        self.assertEqual(self.storage._sanitize_table_name("table.with.dots"), "table_with_dots")
        self.assertEqual(self.storage._sanitize_table_name("mixed-table.name"), "mixed_table_name")
        # Characters beyond hyphen and dot are now also replaced
        self.assertEqual(self.storage._sanitize_table_name("table with spaces"), "table_with_spaces")
        self.assertEqual(self.storage._sanitize_table_name("table@v2!"), "table_v2_")
        # Names starting with a digit get a t_ prefix
        self.assertEqual(self.storage._sanitize_table_name("2024_report"), "t_2024_report")

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

        # Test set — uses run_in_transaction with _try_insert for new keys
        def mock_txn(func, *args, **kwargs):
            func(MagicMock())

        self.mock_database.run_in_transaction.side_effect = mock_txn
        await self.storage.set(key, value)
        self.mock_database.run_in_transaction.assert_called_once()

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
        """set() must auto-create the Blobs table when run_in_transaction raises NotFound."""
        key = "test.txt"
        value = b"hello"

        call_count = 0
        def mock_txn(func, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise exceptions.NotFound("Table not found: Blobs")
            func(MagicMock())

        self.mock_database.run_in_transaction.side_effect = mock_txn
        self.mock_database.update_ddl.return_value = MagicMock()

        await self.storage.set(key, value)

        self.mock_database.update_ddl.assert_called_once()
        ddl = self.mock_database.update_ddl.call_args[0][0][0]
        self.assertIn("CREATE TABLE IF NOT EXISTS `Blobs`", ddl)
        self.assertIn("created_at", ddl)
        self.assertEqual(call_count, 2)

    async def test_set_blob_existing_key_uses_update_transaction(self):
        """set() must use a fresh transaction for UPDATE when INSERT raises AlreadyExists."""
        key = "test.txt"
        value = b"updated"

        call_count = 0
        captured_sqls: list[str] = []

        def mock_txn(func, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_t = MagicMock()
            if call_count == 1:
                # First call: _try_insert raises AlreadyExists
                mock_t.execute_update.side_effect = exceptions.AlreadyExists("exists")
                try:
                    func(mock_t)
                except exceptions.AlreadyExists:
                    raise  # propagate so caller can handle
            else:
                # Second call: _do_update
                func(mock_t)
                captured_sqls.append(mock_t.execute_update.call_args[0][0])

        self.mock_database.run_in_transaction.side_effect = mock_txn

        await self.storage.set(key, value)

        # Two separate run_in_transaction calls: INSERT then UPDATE
        self.assertEqual(call_count, 2)
        # The second SQL must be UPDATE and must NOT touch created_at
        self.assertGreater(len(captured_sqls), 0)
        self.assertIn("UPDATE", captured_sqls[-1])
        self.assertNotIn("created_at", captured_sqls[-1])

    # ------------------------------------------------------------------
    # Blob helpers — gaps from original test suite
    # ------------------------------------------------------------------

    async def test_get_as_string(self):
        """get() without as_bytes should return a decoded string."""
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.read.return_value = [[b"hello world"]]

        result = await self.storage.get("test.txt", as_bytes=False)

        self.assertEqual(result, "hello world")

    async def test_get_base64_encoded_value(self):
        """get() should transparently decode base64-wrapped bytes from Spanner."""
        original = b"raw binary data"
        encoded = base64.b64encode(original)  # what Spanner may return

        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.read.return_value = [[encoded]]

        result = await self.storage.get("test.bin", as_bytes=True)

        self.assertEqual(result, original)

    async def test_get_non_existent_key_returns_none(self):
        """get() should return None when the key is not found."""
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.read.return_value = []  # no rows

        result = await self.storage.get("missing_key")

        self.assertIsNone(result)

    async def test_set_string_value(self):
        """set() should accept string values by encoding them to bytes first."""
        def mock_txn(func, *args, **kwargs):
            func(MagicMock())

        self.mock_database.run_in_transaction.side_effect = mock_txn

        await self.storage.set("test.txt", "hello string")

        self.mock_database.run_in_transaction.assert_called_once()

    async def test_set_uses_parameterized_dml(self):
        """set() must use parameterized DML with FROM_BASE64, not embed value in SQL."""
        captured: dict = {}

        def capture_txn(func, *args, **kwargs):
            mock_t = MagicMock()
            func(mock_t)
            captured["sql"] = mock_t.execute_update.call_args[0][0]
            captured["params"] = mock_t.execute_update.call_args[1]["params"]

        self.mock_database.run_in_transaction.side_effect = capture_txn

        await self.storage.set("key", b"sensitive data")

        self.assertIn("@val", captured["sql"])
        self.assertIn("FROM_BASE64", captured["sql"])
        self.assertIn("val", captured["params"])
        self.assertIn("created_at", captured["sql"])

    # ------------------------------------------------------------------
    # set_table — retry exhaustion
    # ------------------------------------------------------------------

    async def test_set_table_retry_exhausted_raises(self):
        """set_table() should re-raise NotFound after max_retries are exhausted."""
        df = pd.DataFrame({"id": ["1"]})

        mock_batch = MagicMock()
        mock_batch.insert_or_update.side_effect = exceptions.NotFound("Table not found")
        self.mock_database.batch.return_value.__enter__.return_value = mock_batch

        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.execute_sql.return_value = []

        self.mock_database.update_ddl.return_value = MagicMock()

        with self.assertRaises(exceptions.NotFound):
            await self.storage.set_table("FailTable", df)

        # max_retries=2 → loop runs for attempts 0, 1, 2 → 3 insert attempts
        self.assertEqual(mock_batch.insert_or_update.call_count, 3)

    # ------------------------------------------------------------------
    # clear / keys / find / child / get_creation_date
    # ------------------------------------------------------------------

    async def test_clear(self):
        """clear() should execute partitioned DML against the blob table."""
        await self.storage.clear()

        self.mock_database.execute_partitioned_dml.assert_called_once()
        sql = self.mock_database.execute_partitioned_dml.call_args[0][0]
        self.assertIn("DELETE FROM", sql)
        self.assertIn(self.storage._blob_table, sql)

    def test_keys(self):
        """keys() should return all blob key values."""
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.execute_sql.return_value = [["key1"], ["key2"], ["key3"]]

        keys = self.storage.keys()

        self.assertEqual(keys, ["key1", "key2", "key3"])

    def test_find_matches_pattern(self):
        """find() should yield only keys matching the given pattern."""
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.execute_sql.return_value = [
            ["file1.txt"], ["file2.log"], ["file3.txt"]
        ]

        results = list(self.storage.find(re.compile(r".*\.txt$")))

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "file1.txt")
        self.assertEqual(results[1][0], "file3.txt")

    def test_find_with_max_count(self):
        """find() should stop after max_count matches."""
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.execute_sql.return_value = [
            ["f1.txt"], ["f2.txt"], ["f3.txt"], ["f4.txt"]
        ]

        results = list(self.storage.find(re.compile(r".*\.txt$"), max_count=2))

        self.assertEqual(len(results), 2)

    def test_find_with_file_filter(self):
        """find() should apply file_filter against named groups in the pattern."""
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.execute_sql.return_value = [
            ["2024-01-report.txt"],
            ["2023-12-report.txt"],
            ["2024-06-report.txt"],
        ]

        # Pattern captures the year as a named group
        pattern = re.compile(r"(?P<year>\d{4})-\d{2}-report\.txt")
        # Filter: only match year "2024"
        results = list(self.storage.find(pattern, file_filter={"year": "2024"}))

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "2024-01-report.txt")
        self.assertEqual(results[1][0], "2024-06-report.txt")

    def test_child_creates_prefixed_storage(self):
        """child() should return a new SpannerPipelineStorage with a stacked table prefix."""
        child = self.storage.child("reports")

        self.assertIsInstance(child, SpannerPipelineStorage)
        self.assertEqual(child._table_prefix, "reports_")
        self.assertEqual(child._blob_table, "reports_Blobs")

    def test_child_sanitizes_name(self):
        """child() should sanitize hyphens and dots in the child name."""
        child = self.storage.child("my-reports.v2")

        self.assertEqual(child._table_prefix, "my_reports_v2_")

    def test_child_stacks_on_parent_prefix(self):
        """child() should prepend the parent's existing prefix."""
        storage_with_prefix = SpannerPipelineStorage(
            project_id=self.project_id,
            instance_id=self.instance_id,
            database_id=self.database_id,
            table_prefix="app_",
        )
        child = storage_with_prefix.child("reports")

        self.assertEqual(child._table_prefix, "app_reports_")

    def test_child_none_returns_self(self):
        """child(None) should return the same instance."""
        result = self.storage.child(None)
        self.assertIs(result, self.storage)

    async def test_get_creation_date(self):
        """get_creation_date() should return a non-empty string for an existing key."""
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.read.return_value = [[dt]]

        result = await self.storage.get_creation_date("test.txt")

        self.assertIn("2024", result)
        self.assertIn("06", result)
        self.assertIn("15", result)

    async def test_get_creation_date_not_found(self):
        """get_creation_date() should return empty string when key is absent."""
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.read.return_value = []

        result = await self.storage.get_creation_date("missing")

        self.assertEqual(result, "")

    # ------------------------------------------------------------------
    # New tests for Tasks 1–5
    # ------------------------------------------------------------------

    def test_schema_cache_evicts_lru_when_full(self):
        """Schema cache should evict the oldest entry when it reaches max size."""
        from graphrag.storage.spanner_pipeline_storage import _SCHEMA_CACHE_MAX_SIZE

        # Fill the cache to the max size
        for i in range(_SCHEMA_CACHE_MAX_SIZE):
            self.storage._schema_cache[f"table_{i}"] = {"col": "STRING(MAX)"}

        self.assertEqual(len(self.storage._schema_cache), _SCHEMA_CACHE_MAX_SIZE)

        # Mock snapshot so _get_table_schema calls succeed with a real schema
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.execute_sql.return_value = iter([["new_col", "INT64"]])

        # Fetching a brand-new table should evict "table_0" (the oldest)
        self.storage._get_table_schema("brand_new_table")

        self.assertNotIn("table_0", self.storage._schema_cache)
        self.assertIn("brand_new_table", self.storage._schema_cache)
        self.assertEqual(len(self.storage._schema_cache), _SCHEMA_CACHE_MAX_SIZE)

    async def test_load_table_passes_limit_offset(self):
        """load_table() should include LIMIT/OFFSET in the SQL when provided."""
        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot

        mock_results = MagicMock()
        mock_results.__iter__.return_value = iter([[1, "a"]])
        field1 = MagicMock()
        field1.name = "col1"
        field2 = MagicMock()
        field2.name = "col2"
        mock_results.fields = [field1, field2]
        mock_snapshot.execute_sql.return_value = mock_results

        await self.storage.load_table("TestTable", limit=10, offset=5)

        call_args = mock_snapshot.execute_sql.call_args
        sql = call_args[0][0]
        self.assertIn("LIMIT", sql)
        self.assertIn("OFFSET", sql)
        params = call_args[1]["params"]
        self.assertEqual(params["lim"], 10)
        self.assertEqual(params["off"], 5)

    async def test_get_creation_date_reads_created_at_column(self):
        """get_creation_date() should read the 'created_at' column, not 'updated_at'."""
        dt = datetime(2024, 6, 15, 10, 30, 0, tzinfo=timezone.utc)

        mock_snapshot = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_snapshot.read.return_value = [[dt]]

        await self.storage.get_creation_date("test.txt")

        mock_snapshot.read.assert_called_once()
        call_args = mock_snapshot.read.call_args
        # Second positional argument is the list of columns
        columns_arg = call_args[0][1]
        self.assertEqual(columns_arg, ["created_at"])

    async def test_set_insert_sql_includes_created_at(self):
        """INSERT DML must include created_at so it is set on first write."""
        captured_sql = {}

        def capture_txn(func, *args, **kwargs):
            mock_t = MagicMock()
            func(mock_t)
            captured_sql["sql"] = mock_t.execute_update.call_args[0][0]

        self.mock_database.run_in_transaction.side_effect = capture_txn

        await self.storage.set("new_key", b"value")

        self.assertIn("INSERT INTO", captured_sql["sql"])
        self.assertIn("created_at", captured_sql["sql"])
        self.assertIn("updated_at", captured_sql["sql"])

    async def test_set_update_sql_excludes_created_at(self):
        """UPDATE DML must NOT include created_at so it is preserved on re-writes."""
        captured_sqls: list[str] = []
        call_count = 0

        def capture_txn(func, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_t = MagicMock()
            if call_count == 1:
                mock_t.execute_update.side_effect = exceptions.AlreadyExists("exists")
                try:
                    func(mock_t)
                except exceptions.AlreadyExists:
                    raise
            else:
                func(mock_t)
                captured_sqls.append(mock_t.execute_update.call_args[0][0])

        self.mock_database.run_in_transaction.side_effect = capture_txn

        await self.storage.set("existing_key", b"updated_value")

        self.assertEqual(call_count, 2)
        self.assertIn("UPDATE", captured_sqls[-1])
        self.assertNotIn("created_at", captured_sqls[-1])
        self.assertIn("updated_at", captured_sqls[-1])


class TestSafeIdentifier(unittest.TestCase):
    """Tests for the module-level _safe_identifier() guard."""

    def test_valid_simple_name(self):
        self.assertEqual(_safe_identifier("my_table"), "`my_table`")

    def test_valid_with_digits(self):
        self.assertEqual(_safe_identifier("table_1"), "`table_1`")

    def test_rejects_semicolon(self):
        with self.assertRaises(ValueError):
            _safe_identifier("tbl; DROP TABLE foo")

    def test_rejects_space(self):
        with self.assertRaises(ValueError):
            _safe_identifier("my table")

    def test_rejects_hyphen(self):
        """Hyphens must be sanitized before reaching _safe_identifier."""
        with self.assertRaises(ValueError):
            _safe_identifier("my-table")

    def test_rejects_dot(self):
        with self.assertRaises(ValueError):
            _safe_identifier("my.table")

    def test_rejects_leading_digit(self):
        with self.assertRaises(ValueError):
            _safe_identifier("1table")

    def test_rejects_backtick_injection(self):
        with self.assertRaises(ValueError):
            _safe_identifier("`injected`")


class TestInferSpannerType(unittest.TestCase):
    """Unit tests for SpannerPipelineStorage._infer_spanner_type()."""

    def setUp(self):
        self.patcher = patch(
            "graphrag.storage.spanner_pipeline_storage.SpannerResourceManager"
        )
        mock_rm = self.patcher.start()
        mock_rm.get_database.return_value = MagicMock()
        self.storage = SpannerPipelineStorage(
            project_id="p", instance_id="i", database_id="d"
        )

    def tearDown(self):
        self.patcher.stop()

    def _infer(self, series):
        return self.storage._infer_spanner_type(series)

    def test_integer_dtype(self):
        self.assertEqual(self._infer(pd.Series([1, 2, 3])), "INT64")

    def test_float_dtype(self):
        self.assertEqual(self._infer(pd.Series([1.0, 2.0, 3.0])), "FLOAT64")

    def test_bool_dtype(self):
        self.assertEqual(self._infer(pd.Series([True, False, True])), "BOOL")

    def test_datetime_dtype(self):
        s = pd.Series(pd.to_datetime(["2021-01-01", "2021-01-02"]))
        self.assertEqual(self._infer(s), "TIMESTAMP")

    def test_string_dtype(self):
        self.assertEqual(self._infer(pd.Series(["a", "b", "c"])), "STRING(MAX)")

    def test_all_null(self):
        """All-null series should default to STRING(MAX)."""
        self.assertEqual(self._infer(pd.Series([None, None, None])), "STRING(MAX)")

    def test_list_of_strings(self):
        self.assertEqual(
            self._infer(pd.Series([["tag1", "tag2"], ["tag3"]])),
            "ARRAY<STRING(MAX)>",
        )

    def test_list_of_ints(self):
        self.assertEqual(self._infer(pd.Series([[1, 2], [3, 4]])), "ARRAY<INT64>")

    def test_list_of_floats(self):
        self.assertEqual(
            self._infer(pd.Series([[1.0, 2.0], [3.0]])), "ARRAY<FLOAT64>"
        )

    def test_dict_dtype(self):
        self.assertEqual(
            self._infer(pd.Series([{"key": "value"}, {"k2": "v2"}])), "JSON"
        )

    def test_list_of_dicts(self):
        """A list whose elements are dicts should be inferred as JSON."""
        self.assertEqual(self._infer(pd.Series([[{"key": "val"}]])), "JSON")

    def test_all_empty_lists(self):
        """All-empty lists should default to ARRAY<STRING(MAX)>."""
        self.assertEqual(
            self._infer(pd.Series([[], [], []])), "ARRAY<STRING(MAX)>"
        )

    def test_list_with_leading_null(self):
        """None values should be skipped before type inference."""
        self.assertEqual(
            self._infer(pd.Series([None, ["a", "b"]])), "ARRAY<STRING(MAX)>"
        )

    def test_infer_spanner_type_mixed_detects_json(self):
        """Series with mixed dict and str values should resolve to JSON."""
        self.assertEqual(
            self._infer(pd.Series([{}, "text"])), "JSON"
        )

    def test_infer_spanner_type_uses_multiple_samples(self):
        """Series with inconsistent list/dict elements should resolve to JSON."""
        self.assertEqual(
            self._infer(pd.Series([[], [], {}, {}])), "JSON"
        )
