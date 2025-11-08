# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Google Cloud Spanner vector storage implementation."""

import json
import logging
from typing import Any

from google.cloud import spanner
from google.cloud.spanner_v1 import param_types
from google.api_core import exceptions

from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.data_model.types import TextEmbedder
from graphrag.vector_stores.base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

logger = logging.getLogger(__name__)


class SpannerVectorStore(BaseVectorStore):
    """Google Cloud Spanner vector storage implementation."""

    def __init__(
        self, vector_store_schema_config: VectorStoreSchemaConfig, **kwargs: Any
    ) -> None:
        super().__init__(
            vector_store_schema_config=vector_store_schema_config, **kwargs
        )
        # Sanitize index_name for Spanner (replace hyphens with underscores)
        self.index_name = self.index_name.replace("-", "_")

        self._project_id = kwargs.get("project_id")
        self._instance_id = kwargs.get("instance_id")
        self._database_id = kwargs.get("database_id")
        self._credentials = kwargs.get("credentials")

    def connect(self, **kwargs: Any) -> None:
        """Connect to the vector storage."""
        project_id = kwargs.get("project_id", self._project_id)
        instance_id = kwargs.get("instance_id", self._instance_id)
        database_id = kwargs.get("database_id", self._database_id)
        credentials = kwargs.get("credentials", self._credentials)

        if not all([project_id, instance_id, database_id]):
            msg = "project_id, instance_id, and database_id are required for Spanner connection."
            raise ValueError(msg)

        self._client = spanner.Client(project=project_id, credentials=credentials)
        self._instance = self._client.instance(instance_id)
        self._database = self._instance.database(database_id)

    def _create_table_if_not_exists(self) -> None:
        """Create the vector store table and index if they don't exist."""
        table_ddl = f"""CREATE TABLE IF NOT EXISTS `{self.index_name}` (
            `{self.id_field}` STRING(MAX) NOT NULL,
            `{self.text_field}` STRING(MAX),
            `{self.vector_field}` ARRAY<FLOAT64>(vector_length=>{self.vector_size}),
            `{self.attributes_field}` JSON
        ) PRIMARY KEY (`{self.id_field}`)"""

        index_name = f"{self.index_name}_VectorIndex"
        index_ddl = f"""CREATE VECTOR INDEX IF NOT EXISTS `{index_name}`
            ON `{self.index_name}`(`{self.vector_field}`)
            WHERE `{self.vector_field}` IS NOT NULL
            OPTIONS (distance_type = 'COSINE')"""

        logger.info("Creating vector table and index %s if not exists...", self.index_name)
        operation = self._database.update_ddl([table_ddl, index_ddl])
        # Index creation might take longer, so we increase the timeout
        operation.result(timeout=900)
        logger.info("Vector table and index %s created (or already existed).", self.index_name)

    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """Load documents into vector storage."""
        if not documents:
            return

        # Note: 'overwrite=True' here is implemented as UPSERT (insert_or_update).
        # It does NOT truncate the table before loading.

        rows = []
        for doc in documents:
            rows.append(
                (
                    doc.id,
                    doc.text,
                    doc.vector,
                    json.dumps(doc.attributes) if doc.attributes else None,
                )
            )

        try:
            with self._database.batch() as batch:
                batch.insert_or_update(
                    table=self.index_name,
                    columns=(
                        self.id_field,
                        self.text_field,
                        self.vector_field,
                        self.attributes_field,
                    ),
                    values=rows,
                )
        except exceptions.NotFound as e:
            if "Table not found" in str(e):
                logger.info("Table %s not found, attempting to create it.", self.index_name)
                self._create_table_if_not_exists()
                # Retry the insert
                with self._database.batch() as batch:
                    batch.insert_or_update(
                        table=self.index_name,
                        columns=(
                            self.id_field,
                            self.text_field,
                            self.vector_field,
                            self.attributes_field,
                        ),
                        values=rows,
                    )
            else:
                raise

    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """Build a query filter to filter documents by id."""
        self.query_filter = include_ids
        return self.query_filter

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """Perform a vector-based similarity search."""
        where_clause = ""
        params = {"query_vector": query_embedding, "k": k}
        param_types_map = {
            "query_vector": param_types.Array(param_types.FLOAT64),
            "k": param_types.INT64,
        }

        if self.query_filter:
            where_clause = f"WHERE {self.id_field} IN UNNEST(@include_ids)"
            params["include_ids"] = self.query_filter
            if isinstance(self.query_filter[0], int):
                param_types_map["include_ids"] = param_types.Array(param_types.INT64)
            else:
                param_types_map["include_ids"] = param_types.Array(param_types.STRING)

        sql = f"""
            SELECT {self.id_field}, {self.text_field}, {self.vector_field}, {self.attributes_field},
                   COSINE_DISTANCE({self.vector_field}, @query_vector) AS distance
            FROM {self.index_name}
            {where_clause}
            ORDER BY distance
            LIMIT @k
        """

        results = []
        with self._database.snapshot() as snapshot:
            rows = snapshot.execute_sql(sql, params=params, param_types=param_types_map)
            for row in rows:
                # row[0]: id, row[1]: text, row[2]: vector, row[3]: attributes, row[4]: distance
                doc_id = row[0]
                text = row[1]
                vector = row[2]
                attributes_json = row[3]
                distance = row[4]

                attributes = attributes_json if attributes_json else {}

                results.append(
                    VectorStoreSearchResult(
                        document=VectorStoreDocument(
                            id=doc_id,
                            text=text,
                            vector=vector,
                            attributes=attributes,
                        ),
                        score=1 - distance,
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
        """Search for a document by id."""
        params = {"id": id}
        param_type = param_types.STRING
        if isinstance(id, int):
            param_type = param_types.INT64

        sql = f"""
            SELECT {self.id_field}, {self.text_field}, {self.vector_field}, {self.attributes_field}
            FROM {self.index_name}
            WHERE {self.id_field} = @id
        """

        with self._database.snapshot() as snapshot:
            rows = list(
                snapshot.execute_sql(sql, params=params, param_types={"id": param_type})
            )
            if rows:
                row = rows[0]
                attributes = row[3] if row[3] else {}
                return VectorStoreDocument(
                    id=row[0],
                    text=row[1],
                    vector=row[2],
                    attributes=attributes,
                )

        return VectorStoreDocument(id=id, text=None, vector=None)

    def close(self) -> None:
        """Close the Spanner client."""
        if hasattr(self, "_client") and self._client:
            self._client.close()
                