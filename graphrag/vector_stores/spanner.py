# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Google Cloud Spanner vector storage implementation."""

import json
import logging
import re
from typing import Any

from google.cloud import spanner
from google.api_core import exceptions
from google.cloud.spanner_v1 import param_types

from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.data_model.types import TextEmbedder
from graphrag.utils.spanner_resource_manager import SpannerResourceManager
from graphrag.vector_stores.base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL-injection guard (mirrors spanner_pipeline_storage._safe_identifier)
# ---------------------------------------------------------------------------

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_identifier(name: str) -> str:
    """Validate and backtick-quote a Spanner identifier to prevent SQL injection."""
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Unsafe Spanner identifier {name!r}: only letters, digits, and "
            "underscores are permitted, and the name must not start with a digit."
        )
    return f"`{name}`"


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

        logger.debug("Creating SpannerVectorStore: obj_id=%s, index=%s", id(self), self.index_name)

        self._project_id = kwargs.get("project_id")
        self._instance_id = kwargs.get("instance_id")
        self._database_id = kwargs.get("database_id")
        self._credentials = kwargs.get("credentials")
        self._database = None

    def connect(self, **kwargs: Any) -> None:
        """Connect to the vector storage."""
        project_id = kwargs.get("project_id", self._project_id)
        instance_id = kwargs.get("instance_id", self._instance_id)
        database_id = kwargs.get("database_id", self._database_id)
        credentials = kwargs.get("credentials", self._credentials)

        if not all([project_id, instance_id, database_id]):
            msg = "project_id, instance_id, and database_id are required for Spanner connection."
            raise ValueError(msg)

        # Use the shared resource manager
        # Ensure we don't leak a reference if connect is called multiple times
        if self._database:
             SpannerResourceManager.release_database(self._database)
             
        self._database = SpannerResourceManager.get_database(
            project_id=project_id,
            instance_id=instance_id,
            database_id=database_id,
            credentials=credentials
        )
        # We don't need self._client or self._instance anymore, as they are managed internally by SpannerResourceManager

    def _create_table_if_not_exists(self) -> None:
        """Create the vector store table and index if they don't exist."""
        _t = _safe_identifier(self.index_name)
        _id = _safe_identifier(self.id_field)
        _text = _safe_identifier(self.text_field)
        _vec = _safe_identifier(self.vector_field)
        _attrs = _safe_identifier(self.attributes_field)

        table_ddl = f"""CREATE TABLE IF NOT EXISTS {_t} (
            {_id} STRING(MAX) NOT NULL,
            {_text} STRING(MAX),
            {_vec} ARRAY<FLOAT64>(vector_length=>{self.vector_size}),
            {_attrs} JSON
        ) PRIMARY KEY ({_id})"""

        vi = _safe_identifier(f"{self.index_name}_VectorIndex")
        index_ddl = f"""CREATE VECTOR INDEX IF NOT EXISTS {vi}
            ON {_t}({_vec})
            WHERE {_vec} IS NOT NULL
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

        _t = _safe_identifier(self.index_name)
        _id = _safe_identifier(self.id_field)
        _text = _safe_identifier(self.text_field)
        _vec = _safe_identifier(self.vector_field)
        _attrs = _safe_identifier(self.attributes_field)

        if self.query_filter:
            where_clause = f"WHERE {_id} IN UNNEST(@include_ids)"
            params["include_ids"] = self.query_filter
            if isinstance(self.query_filter[0], int):
                param_types_map["include_ids"] = param_types.Array(param_types.INT64)
            else:
                param_types_map["include_ids"] = param_types.Array(param_types.STRING)

        sql = f"""
            SELECT {_id}, {_text}, {_vec}, {_attrs},
                   COSINE_DISTANCE({_vec}, @query_vector) AS distance
            FROM {_t}
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

        _t = _safe_identifier(self.index_name)
        _id = _safe_identifier(self.id_field)
        _text = _safe_identifier(self.text_field)
        _vec = _safe_identifier(self.vector_field)
        _attrs = _safe_identifier(self.attributes_field)

        sql = f"""
            SELECT {_id}, {_text}, {_vec}, {_attrs}
            FROM {_t}
            WHERE {_id} = @id
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
        """Release the Spanner database resource."""
        if hasattr(self, "_database") and self._database:
            logger.debug("Releasing SpannerVectorStore database: obj_id=%s", id(self))
            SpannerResourceManager.release_database(self._database)
            self._database = None