# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for SpannerVectorStore (v3 graphrag-vectors package)."""

import json
import unittest
from unittest.mock import MagicMock, patch

from graphrag_vectors.spanner import SpannerVectorStore, _safe_identifier
from graphrag_vectors.vector_store import VectorStoreDocument


class TestSafeIdentifier(unittest.TestCase):
    def test_valid_identifier(self):
        self.assertEqual(_safe_identifier("my_table"), "`my_table`")

    def test_identifier_with_letters_and_digits(self):
        self.assertEqual(_safe_identifier("Table1"), "`Table1`")

    def test_invalid_identifier_raises(self):
        with self.assertRaises(ValueError):
            _safe_identifier("bad-name")

    def test_identifier_starting_with_digit_raises(self):
        with self.assertRaises(ValueError):
            _safe_identifier("1bad")

    def test_identifier_with_spaces_raises(self):
        with self.assertRaises(ValueError):
            _safe_identifier("bad name")


class TestSpannerVectorStore(unittest.TestCase):
    def setUp(self):
        self.project_id = "test-project"
        self.instance_id = "test-instance"
        self.database_id = "test-database"
        self.index_name = "test_table"
        self.mock_database = MagicMock()

        self.srm_patcher = patch(
            "graphrag_vectors.spanner.SpannerResourceManager"
        )
        # SpannerResourceManager is imported inside connect() at runtime
        # so we patch it at module level
        import graphrag_vectors.spanner as spanner_mod

        self._srm_patcher = patch.object(
            spanner_mod, "SpannerResourceManager" if hasattr(spanner_mod, "SpannerResourceManager") else "_srm_placeholder",
            create=True,
        )

        # Simpler approach: patch inside connect()
        self.mock_resource_manager = MagicMock()
        self.mock_resource_manager.get_database.return_value = self.mock_database

        self.store = SpannerVectorStore(
            index_name=self.index_name,
            vector_size=3,
            project_id=self.project_id,
            instance_id=self.instance_id,
            database_id=self.database_id,
        )
        # Manually inject the mock database (bypassing connect())
        self.store._database = self.mock_database

    # ------------------------------------------------------------------
    # _safe_identifier (duplicated here for in-class access)
    # ------------------------------------------------------------------

    def test_index_name_hyphen_replaced(self):
        """index_name with hyphens should be sanitized to underscores."""
        store = SpannerVectorStore(
            index_name="my-index",
            project_id="p", instance_id="i", database_id="d",
        )
        self.assertEqual(store.index_name, "my_index")

    # ------------------------------------------------------------------
    # load_documents
    # ------------------------------------------------------------------

    def test_load_documents_inserts_rows(self):
        docs = [
            VectorStoreDocument(
                id="1", vector=[0.1, 0.2, 0.3], data={"text": "doc1", "attr": "x"}
            ),
            VectorStoreDocument(
                id="2", vector=[0.4, 0.5, 0.6], data={}
            ),
        ]
        mock_batch = MagicMock()
        self.mock_database.batch.return_value.__enter__ = MagicMock(return_value=mock_batch)
        self.mock_database.batch.return_value.__exit__ = MagicMock(return_value=False)

        self.store.load_documents(docs)

        # Should have called execute_partitioned_dml (DELETE) then insert_or_update
        self.mock_database.execute_partitioned_dml.assert_called_once()
        mock_batch.insert_or_update.assert_called_once()

        call_args = mock_batch.insert_or_update.call_args
        self.assertEqual(call_args.kwargs["table"], self.index_name)
        self.assertIn("data", call_args.kwargs["columns"])
        self.assertIn("vector", call_args.kwargs["columns"])

        rows = call_args.kwargs["values"]
        self.assertEqual(len(rows), 2)
        # First row: id=1, data should be JSON
        self.assertEqual(rows[0][0], "1")
        data_col = rows[0][2]
        self.assertIsNotNone(data_col)
        parsed = json.loads(data_col)
        self.assertEqual(parsed["text"], "doc1")

    def test_load_documents_empty(self):
        self.store.load_documents([])
        self.mock_database.execute_partitioned_dml.assert_not_called()

    def test_load_documents_skips_none_vector(self):
        docs = [
            VectorStoreDocument(id="1", vector=None, data={"text": "no vector"}),
        ]
        mock_batch = MagicMock()
        self.mock_database.batch.return_value.__enter__ = MagicMock(return_value=mock_batch)
        self.mock_database.batch.return_value.__exit__ = MagicMock(return_value=False)

        self.store.load_documents(docs)

        # After DELETE, insert_or_update called with zero rows (skipped)
        if mock_batch.insert_or_update.called:
            rows = mock_batch.insert_or_update.call_args.kwargs["values"]
            self.assertEqual(len(rows), 0)

    # ------------------------------------------------------------------
    # count
    # ------------------------------------------------------------------

    def test_count_returns_row_count(self):
        mock_snapshot = MagicMock()
        mock_snapshot.execute_sql.return_value = [(42,)]
        self.mock_database.snapshot.return_value.__enter__ = MagicMock(
            return_value=mock_snapshot
        )
        self.mock_database.snapshot.return_value.__exit__ = MagicMock(return_value=False)

        result = self.store.count()
        self.assertEqual(result, 42)

    # ------------------------------------------------------------------
    # remove
    # ------------------------------------------------------------------

    def test_remove_calls_batch_delete(self):
        mock_batch = MagicMock()
        self.mock_database.batch.return_value.__enter__ = MagicMock(return_value=mock_batch)
        self.mock_database.batch.return_value.__exit__ = MagicMock(return_value=False)

        self.store.remove(["id1", "id2"])
        mock_batch.delete.assert_called_once()

    def test_remove_empty_list_is_noop(self):
        self.store.remove([])
        self.mock_database.batch.assert_not_called()

    # ------------------------------------------------------------------
    # search_by_id
    # ------------------------------------------------------------------

    def test_search_by_id_found(self):
        mock_snapshot = MagicMock()
        # Real Spanner client returns JSON columns as dict, not a JSON string
        data_payload = {"text": "hello"}
        mock_snapshot.execute_sql.return_value = [
            ("1", [0.1, 0.2, 0.3], data_payload, "2024-01-01", None)
        ]
        self.mock_database.snapshot.return_value.__enter__ = MagicMock(
            return_value=mock_snapshot
        )
        self.mock_database.snapshot.return_value.__exit__ = MagicMock(return_value=False)

        doc = self.store.search_by_id("1")
        self.assertEqual(doc.id, "1")
        self.assertEqual(doc.vector, [0.1, 0.2, 0.3])
        self.assertEqual(doc.data.get("text"), "hello")

    def test_search_by_id_not_found(self):
        mock_snapshot = MagicMock()
        mock_snapshot.execute_sql.return_value = []
        self.mock_database.snapshot.return_value.__enter__ = MagicMock(
            return_value=mock_snapshot
        )
        self.mock_database.snapshot.return_value.__exit__ = MagicMock(return_value=False)

        doc = self.store.search_by_id("missing")
        self.assertEqual(doc.id, "missing")
        self.assertIsNone(doc.vector)

    # ------------------------------------------------------------------
    # similarity_search_by_vector
    # ------------------------------------------------------------------

    def test_similarity_search_returns_results(self):
        mock_snapshot = MagicMock()
        # Real Spanner client returns JSON columns as dict
        data_payload = {"category": "A"}
        mock_snapshot.execute_sql.return_value = [
            ("1", [0.1, 0.2, 0.3], data_payload, "2024-01-01", None, 0.1),
            ("2", [0.4, 0.5, 0.6], None, "2024-01-02", None, 0.3),
        ]
        self.mock_database.snapshot.return_value.__enter__ = MagicMock(
            return_value=mock_snapshot
        )
        self.mock_database.snapshot.return_value.__exit__ = MagicMock(return_value=False)

        results = self.store.similarity_search_by_vector([0.1, 0.2, 0.3], k=5)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].score, 0.9)
        self.assertEqual(results[0].document.id, "1")

    def test_similarity_search_with_select(self):
        mock_snapshot = MagicMock()
        # Real Spanner client returns JSON columns as dict
        data_payload = {"text": "hello", "extra": "skip"}
        mock_snapshot.execute_sql.return_value = [
            ("1", [0.1, 0.2, 0.3], data_payload, None, None, 0.2),
        ]
        self.mock_database.snapshot.return_value.__enter__ = MagicMock(
            return_value=mock_snapshot
        )
        self.mock_database.snapshot.return_value.__exit__ = MagicMock(return_value=False)

        results = self.store.similarity_search_by_vector(
            [0.1, 0.2, 0.3], k=5, select=["text"]
        )
        self.assertIn("text", results[0].document.data)
        self.assertNotIn("extra", results[0].document.data)

    def test_similarity_search_exclude_vectors(self):
        mock_snapshot = MagicMock()
        mock_snapshot.execute_sql.return_value = [
            ("1", [0.1, 0.2, 0.3], None, None, None, 0.1),
        ]
        self.mock_database.snapshot.return_value.__enter__ = MagicMock(
            return_value=mock_snapshot
        )
        self.mock_database.snapshot.return_value.__exit__ = MagicMock(return_value=False)

        results = self.store.similarity_search_by_vector(
            [0.1, 0.2, 0.3], k=5, include_vectors=False
        )
        self.assertIsNone(results[0].document.vector)

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def test_update_calls_insert_or_update(self):
        mock_batch = MagicMock()
        self.mock_database.batch.return_value.__enter__ = MagicMock(return_value=mock_batch)
        self.mock_database.batch.return_value.__exit__ = MagicMock(return_value=False)

        doc = VectorStoreDocument(id="1", vector=[0.1, 0.2, 0.3], data={"text": "updated"})
        self.store.update(doc)

        mock_batch.insert_or_update.assert_called_once()
