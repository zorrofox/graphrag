# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Google Vertex AI Vector Search backend for GraphRAG."""

import logging
from typing import Any

from google.cloud import aiplatform

from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.data_model.types import TextEmbedder
from graphrag.vector_stores.base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

logger = logging.getLogger(__name__)


class VertexAIVectorStore(BaseVectorStore):
    """Google Vertex AI Vector Search backend for GraphRAG.

    Uses ``MatchingEngineIndex`` / ``MatchingEngineIndexEndpoint`` from the
    ``google-cloud-aiplatform`` SDK.  The Index and IndexEndpoint must already
    exist; this class does **not** create them.
    """

    def __init__(
        self, vector_store_schema_config: VectorStoreSchemaConfig, **kwargs: Any
    ) -> None:
        super().__init__(
            vector_store_schema_config=vector_store_schema_config, **kwargs
        )
        self._project_id: str | None = kwargs.get("project_id")
        self._location: str | None = kwargs.get("location")
        self._index_id: str | None = kwargs.get("index_id")
        self._index_endpoint_id: str | None = kwargs.get("index_endpoint_id")
        self._deployed_index_id: str | None = kwargs.get("deployed_index_id")
        self._index: Any = None
        self._index_endpoint: Any = None

    def connect(self, **kwargs: Any) -> None:
        """Connect to Vertex AI Vector Search.

        Parameters
        ----------
        project_id:
            GCP project identifier.
        location:
            GCP region, e.g. ``"us-central1"``.
        index_id:
            Full resource name or numeric ID of the MatchingEngineIndex.
        index_endpoint_id:
            Full resource name or numeric ID of the MatchingEngineIndexEndpoint.
        deployed_index_id:
            The deployed index ID registered on the endpoint.
        """
        project_id = kwargs.get("project_id", self._project_id)
        location = kwargs.get("location", self._location)
        index_id = kwargs.get("index_id", self._index_id)
        index_endpoint_id = kwargs.get("index_endpoint_id", self._index_endpoint_id)
        self._deployed_index_id = kwargs.get("deployed_index_id", self._deployed_index_id)

        if not all([project_id, location, index_id, index_endpoint_id, self._deployed_index_id]):
            msg = (
                "project_id, location, index_id, index_endpoint_id, and "
                "deployed_index_id are all required for Vertex AI Vector Search."
            )
            raise ValueError(msg)

        logger.info(
            "Connecting to Vertex AI Vector Search (project=%s, location=%s, index=%s)",
            project_id,
            location,
            index_id,
        )
        aiplatform.init(project=project_id, location=location)
        self._index = aiplatform.MatchingEngineIndex(index_name=index_id)
        self._index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=index_endpoint_id
        )

    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True  # noqa: ARG002
    ) -> None:
        """Upsert documents into the Vertex AI index.

        The *overwrite* flag is accepted for interface compatibility but Vertex
        AI Vector Search only supports upsert semantics; a full replacement
        would require deleting all existing datapoints first, which is not
        implemented here.
        """
        if not documents:
            return

        datapoints = []
        for doc in documents:
            if doc.vector is None:
                continue
            restricts: list[Any] = []
            if doc.text:
                restricts = [
                    aiplatform.MatchingEngineIndex.Datapoint.Restriction(
                        namespace="text",
                        allow_list=[doc.text],
                    )
                ]
            datapoints.append(
                aiplatform.MatchingEngineIndex.Datapoint(
                    datapoint_id=str(doc.id),
                    feature_vector=doc.vector,
                    restricts=restricts,
                )
            )

        if datapoints:
            logger.info("Upserting %d datapoints into Vertex AI index.", len(datapoints))
            self._index.upsert_datapoints(datapoints=datapoints)

    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Store an ID filter to apply on the next similarity search.

        Vertex AI Vector Search supports ``restricts``-based pre-filtering but
        not arbitrary ID allow-lists via the public ``find_neighbors`` API.
        The filter is stored here and cleared after each search call so that
        callers can still set it; however, the actual server-side filtering is
        a no-op for this backend.
        """
        self.query_filter = include_ids
        return self.query_filter

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform an approximate nearest-neighbour search by embedding vector."""
        # Consume and immediately clear the filter so it does not leak into
        # subsequent searches that do not call filter_by_id().
        self.query_filter = None

        response = self._index_endpoint.find_neighbors(
            deployed_index_id=self._deployed_index_id,
            queries=[query_embedding],
            num_neighbors=k,
        )

        results: list[VectorStoreSearchResult] = []
        for neighbor in response[0] if response else []:
            results.append(
                VectorStoreSearchResult(
                    document=VectorStoreDocument(
                        id=neighbor.id,
                        text=None,
                        vector=None,
                    ),
                    score=1.0 - float(neighbor.distance),
                )
            )
        return results

    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a similarity search using a given input text."""
        query_embedding = text_embedder(text)
        if query_embedding:
            return self.similarity_search_by_vector(query_embedding, k)
        return []

    def search_by_id(self, id: str) -> VectorStoreDocument:
        """Look up a document by its datapoint ID.

        Vertex AI Vector Search does not expose a public direct-ID lookup via
        the SDK.  A stub document containing only the requested ID is returned.
        """
        logger.debug(
            "search_by_id is not supported for Vertex AI Vector Search; returning stub for id=%s",
            id,
        )
        return VectorStoreDocument(id=id, text=None, vector=None)
