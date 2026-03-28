# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for the Vertex AI Vector Search backend."""

import unittest
from unittest.mock import MagicMock, patch

from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.vector_stores.base import VectorStoreDocument
from graphrag.vector_stores.vertexai import VertexAIVectorStore


class TestVertexAIVectorStore(unittest.TestCase):
    """Tests for VertexAIVectorStore."""

    def setUp(self):
        self.config = VectorStoreSchemaConfig(index_name="test_index", vector_size=3)
        self.store = VertexAIVectorStore(vector_store_schema_config=self.config)

    @patch("graphrag.vector_stores.vertexai.aiplatform")
    def test_connect_initializes_aiplatform(self, mock_aiplatform):
        mock_aiplatform.MatchingEngineIndex.return_value = MagicMock()
        mock_aiplatform.MatchingEngineIndexEndpoint.return_value = MagicMock()
        self.store.connect(
            project_id="proj",
            location="us-central1",
            index_id="index-1",
            index_endpoint_id="ep-1",
            deployed_index_id="dep-1",
        )
        mock_aiplatform.init.assert_called_once_with(
            project="proj", location="us-central1"
        )
        mock_aiplatform.MatchingEngineIndex.assert_called_once_with(
            index_name="index-1"
        )
        mock_aiplatform.MatchingEngineIndexEndpoint.assert_called_once_with(
            index_endpoint_name="ep-1"
        )

    @patch("graphrag.vector_stores.vertexai.aiplatform")
    def test_connect_raises_on_missing_params(self, mock_aiplatform):
        with self.assertRaises(ValueError):
            self.store.connect(project_id="proj", location="us-central1")

    @patch("graphrag.vector_stores.vertexai.aiplatform")
    def test_load_documents_calls_upsert(self, mock_aiplatform):
        mock_index = MagicMock()
        mock_aiplatform.MatchingEngineIndex.return_value = mock_index
        mock_aiplatform.MatchingEngineIndexEndpoint.return_value = MagicMock()
        self.store.connect(
            project_id="p",
            location="l",
            index_id="i",
            index_endpoint_id="e",
            deployed_index_id="d",
        )
        docs = [VectorStoreDocument(id="1", text="hello", vector=[0.1, 0.2, 0.3])]
        self.store.load_documents(docs)
        mock_index.upsert_datapoints.assert_called_once()

    @patch("graphrag.vector_stores.vertexai.aiplatform")
    def test_load_documents_skips_docs_without_vector(self, mock_aiplatform):
        mock_index = MagicMock()
        mock_aiplatform.MatchingEngineIndex.return_value = mock_index
        mock_aiplatform.MatchingEngineIndexEndpoint.return_value = MagicMock()
        self.store.connect(
            project_id="p",
            location="l",
            index_id="i",
            index_endpoint_id="e",
            deployed_index_id="d",
        )
        docs = [VectorStoreDocument(id="1", text="hello", vector=None)]
        self.store.load_documents(docs)
        mock_index.upsert_datapoints.assert_not_called()

    @patch("graphrag.vector_stores.vertexai.aiplatform")
    def test_similarity_search(self, mock_aiplatform):
        mock_endpoint = MagicMock()
        neighbor = MagicMock()
        neighbor.id = "doc1"
        neighbor.distance = 0.1
        mock_endpoint.find_neighbors.return_value = [[neighbor]]
        mock_aiplatform.MatchingEngineIndex.return_value = MagicMock()
        mock_aiplatform.MatchingEngineIndexEndpoint.return_value = mock_endpoint
        self.store.connect(
            project_id="p",
            location="l",
            index_id="i",
            index_endpoint_id="e",
            deployed_index_id="dep",
        )
        results = self.store.similarity_search_by_vector([0.1, 0.2, 0.3], k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document.id, "doc1")
        self.assertAlmostEqual(results[0].score, 0.9)
        mock_endpoint.find_neighbors.assert_called_once_with(
            deployed_index_id="dep",
            queries=[[0.1, 0.2, 0.3]],
            num_neighbors=5,
        )

    @patch("graphrag.vector_stores.vertexai.aiplatform")
    def test_similarity_search_empty_response(self, mock_aiplatform):
        mock_endpoint = MagicMock()
        mock_endpoint.find_neighbors.return_value = []
        mock_aiplatform.MatchingEngineIndex.return_value = MagicMock()
        mock_aiplatform.MatchingEngineIndexEndpoint.return_value = mock_endpoint
        self.store.connect(
            project_id="p",
            location="l",
            index_id="i",
            index_endpoint_id="e",
            deployed_index_id="dep",
        )
        results = self.store.similarity_search_by_vector([0.1, 0.2, 0.3])
        self.assertEqual(results, [])

    def test_load_documents_empty(self):
        self.store._index = MagicMock()
        self.store.load_documents([])
        self.store._index.upsert_datapoints.assert_not_called()

    def test_query_filter_cleared_after_search(self):
        """After a search the filter must be None."""
        self.store._index_endpoint = MagicMock()
        self.store._index_endpoint.find_neighbors.return_value = [[]]
        self.store._deployed_index_id = "dep"
        self.store.filter_by_id(["1"])
        self.assertIsNotNone(self.store.query_filter)
        self.store.similarity_search_by_vector([0.1, 0.2, 0.3])
        self.assertIsNone(self.store.query_filter)

    def test_filter_by_id_sets_filter(self):
        result = self.store.filter_by_id(["a", "b"])
        self.assertEqual(result, ["a", "b"])
        self.assertEqual(self.store.query_filter, ["a", "b"])

    def test_search_by_id_returns_stub(self):
        doc = self.store.search_by_id("abc")
        self.assertEqual(doc.id, "abc")
        self.assertIsNone(doc.text)
        self.assertIsNone(doc.vector)

    @patch("graphrag.vector_stores.vertexai.aiplatform")
    def test_similarity_search_by_text(self, mock_aiplatform):
        mock_endpoint = MagicMock()
        neighbor = MagicMock()
        neighbor.id = "doc2"
        neighbor.distance = 0.2
        mock_endpoint.find_neighbors.return_value = [[neighbor]]
        mock_aiplatform.MatchingEngineIndex.return_value = MagicMock()
        mock_aiplatform.MatchingEngineIndexEndpoint.return_value = mock_endpoint
        self.store.connect(
            project_id="p",
            location="l",
            index_id="i",
            index_endpoint_id="e",
            deployed_index_id="dep",
        )
        text_embedder = MagicMock(return_value=[0.1, 0.2, 0.3])
        results = self.store.similarity_search_by_text("hello", text_embedder, k=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document.id, "doc2")
        text_embedder.assert_called_once_with("hello")

    @patch("graphrag.vector_stores.vertexai.aiplatform")
    def test_similarity_search_by_text_no_embedding(self, mock_aiplatform):
        self.store._index_endpoint = MagicMock()
        text_embedder = MagicMock(return_value=None)
        results = self.store.similarity_search_by_text("hello", text_embedder)
        self.assertEqual(results, [])
        self.store._index_endpoint.find_neighbors.assert_not_called()

    def test_factory_registration(self):
        """VertexAI must be registered in VectorStoreFactory."""
        from graphrag.vector_stores.factory import VectorStoreFactory

        self.assertIn("vertexai", VectorStoreFactory.get_vector_store_types())


if __name__ == "__main__":
    unittest.main()
