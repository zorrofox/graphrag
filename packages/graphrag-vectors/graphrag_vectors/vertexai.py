# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Google Vertex AI Vector Search backend for graphrag-vectors v3."""

import logging
from typing import Any

from google.cloud import aiplatform

from graphrag_vectors.filtering import FilterExpr
from graphrag_vectors.vector_store import (
    VectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

logger = logging.getLogger(__name__)


class VertexAIVectorStore(VectorStore):
    """Google Vertex AI Vector Search backend.

    Uses MatchingEngineIndex / MatchingEngineIndexEndpoint from the
    google-cloud-aiplatform SDK.  The Index and IndexEndpoint must already
    exist; this class does **not** create them.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._project_id: str | None = kwargs.get("project_id")
        self._location: str | None = kwargs.get("location")
        self._index_id: str | None = kwargs.get("index_id")
        self._index_endpoint_id: str | None = kwargs.get("index_endpoint_id")
        self._deployed_index_id: str | None = kwargs.get("deployed_index_id")
        self._index: Any = None
        self._index_endpoint: Any = None

    def connect(self) -> None:
        """Connect to Vertex AI Vector Search."""
        if not all([
            self._project_id,
            self._location,
            self._index_id,
            self._index_endpoint_id,
            self._deployed_index_id,
        ]):
            msg = (
                "project_id, location, index_id, index_endpoint_id, and "
                "deployed_index_id are all required for Vertex AI Vector Search."
            )
            raise ValueError(msg)

        logger.info(
            "Connecting to Vertex AI Vector Search (project=%s, location=%s, index=%s)",
            self._project_id,
            self._location,
            self._index_id,
        )
        aiplatform.init(project=self._project_id, location=self._location)
        self._index = aiplatform.MatchingEngineIndex(index_name=self._index_id)
        self._index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=self._index_endpoint_id
        )

    def create_index(self) -> None:
        """No-op: Vertex AI indexes must be pre-created in the GCP console."""
        logger.info(
            "create_index() is a no-op for VertexAIVectorStore — "
            "the index must be pre-created in the GCP console."
        )

    def load_documents(self, documents: list[VectorStoreDocument]) -> None:
        """Upsert documents into the Vertex AI index."""
        if not documents:
            return

        datapoints = []
        for doc in documents:
            self._prepare_document(doc)
            if doc.vector is None:
                continue
            restricts: list[Any] = []
            # Store text from data dict as a restriction namespace if present
            text = doc.data.get("text") if doc.data else None
            if text:
                restricts = [
                    aiplatform.MatchingEngineIndex.Datapoint.Restriction(
                        namespace="text",
                        allow_list=[str(text)],
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
            logger.info(
                "Upserting %d datapoints into Vertex AI index.", len(datapoints)
            )
            self._index.upsert_datapoints(datapoints=datapoints)

    def similarity_search_by_vector(
        self,
        query_embedding: list[float],
        k: int = 10,
        select: list[str] | None = None,
        filters: FilterExpr | None = None,
        include_vectors: bool = True,
    ) -> list[VectorStoreSearchResult]:
        """Perform approximate nearest-neighbour search by embedding vector."""
        response = self._index_endpoint.find_neighbors(
            deployed_index_id=self._deployed_index_id,
            queries=[query_embedding],
            num_neighbors=k,
        )

        results: list[VectorStoreSearchResult] = []
        for neighbor in response[0] if response else []:
            doc = VectorStoreDocument(
                id=neighbor.id,
                vector=None,
                data={},
            )
            result = VectorStoreSearchResult(
                document=doc,
                score=1.0 - float(neighbor.distance),
            )
            if filters is None or filters.evaluate(doc):
                results.append(result)

        return results

    def search_by_id(
        self,
        id: str,
        select: list[str] | None = None,
        include_vectors: bool = True,
    ) -> VectorStoreDocument:
        """Look up a document by its datapoint ID.

        Vertex AI Vector Search does not expose a public direct-ID lookup via
        the SDK — a stub document containing only the ID is returned.
        """
        logger.debug(
            "search_by_id is not supported for Vertex AI Vector Search; "
            "returning stub for id=%s",
            id,
        )
        return VectorStoreDocument(id=id, vector=None, data={})

    def count(self) -> int:
        """Return the total number of documents.

        Vertex AI Vector Search does not expose a document count API.
        """
        logger.debug(
            "count() is not supported for VertexAIVectorStore; returning 0"
        )
        return 0

    def remove(self, ids: list[str]) -> None:
        """Remove documents by datapoint ID."""
        if not ids:
            return
        logger.info("Removing %d datapoints from Vertex AI index.", len(ids))
        self._index.remove_datapoints(datapoint_ids=ids)

    def update(self, document: VectorStoreDocument) -> None:
        """Update a document by upserting it."""
        self.load_documents([document])
