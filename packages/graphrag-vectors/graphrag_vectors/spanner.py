# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Google Cloud Spanner vector storage implementation for graphrag-vectors v3."""

import json
import logging
import re
from typing import Any

from google.api_core import exceptions
from google.cloud import spanner
from google.cloud.spanner_v1 import param_types

from graphrag_vectors.filtering import FilterExpr
from graphrag_vectors.vector_store import (
    VectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)

logger = logging.getLogger(__name__)

_DDL_TIMEOUT_SECONDS: int = 600
_VECTOR_INDEX_DDL_TIMEOUT_SECONDS: int = 900

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_identifier(name: str) -> str:
    """Validate and backtick-quote a Spanner identifier to prevent SQL injection."""
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Unsafe Spanner identifier {name!r}: only letters, digits, and "
            "underscores are permitted, and the name must not start with a digit."
        )
    return f"`{name}`"


class SpannerVectorStore(VectorStore):
    """Google Cloud Spanner vector storage implementation.

    Table schema:
        id          STRING(MAX) NOT NULL  (primary key)
        vector      ARRAY<FLOAT64>(vector_length=>N)
        data        JSON                  (all non-vector document fields)
        create_date STRING(MAX)
        update_date STRING(MAX)
    """

    # Name of the JSON column that holds all document.data fields.
    DATA_FIELD = "data"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Sanitize index_name for Spanner (replace hyphens with underscores)
        self.index_name = self.index_name.replace("-", "_")

        logger.debug(
            "Creating SpannerVectorStore: obj_id=%s, index=%s", id(self), self.index_name
        )

        self._project_id: str | None = kwargs.get("project_id")
        self._instance_id: str | None = kwargs.get("instance_id")
        self._database_id: str | None = kwargs.get("database_id")
        self._credentials: Any = kwargs.get("credentials")
        self._database: Any = None

    def connect(self) -> None:
        """Connect to the Spanner vector store."""
        # Import here to avoid circular imports and to enable lazy loading.
        from graphrag_storage.spanner_resource_manager import (
            SpannerResourceManager,  # type: ignore[import]
        )

        if not all([self._project_id, self._instance_id, self._database_id]):
            msg = "project_id, instance_id, and database_id are required for Spanner connection."
            raise ValueError(msg)

        if self._database:
            SpannerResourceManager.release_database(self._database)

        self._database = SpannerResourceManager.get_database(
            project_id=self._project_id,
            instance_id=self._instance_id,
            database_id=self._database_id,
            credentials=self._credentials,
        )

    def create_index(self) -> None:
        """Create the Spanner table and vector index if they don't exist."""
        _t = _safe_identifier(self.index_name)
        _id = _safe_identifier(self.id_field)
        _vec = _safe_identifier(self.vector_field)
        _cdate = _safe_identifier(self.create_date_field)
        _udate = _safe_identifier(self.update_date_field)
        _data = _safe_identifier(self.DATA_FIELD)

        table_ddl = (
            f"CREATE TABLE IF NOT EXISTS {_t} (\n"
            f"    {_id} STRING(MAX) NOT NULL,\n"
            f"    {_vec} ARRAY<FLOAT64>(vector_length=>{self.vector_size}),\n"
            f"    {_data} JSON,\n"
            f"    {_cdate} STRING(MAX),\n"
            f"    {_udate} STRING(MAX)\n"
            f") PRIMARY KEY ({_id})"
        )
        vi = _safe_identifier(f"{self.index_name}_VectorIndex")
        index_ddl = (
            f"CREATE VECTOR INDEX IF NOT EXISTS {vi}\n"
            f"    ON {_t}({_vec})\n"
            f"    WHERE {_vec} IS NOT NULL\n"
            "    OPTIONS (distance_type = 'COSINE')"
        )

        logger.info(
            "Creating vector table and index %s if not exists...", self.index_name
        )
        operation = self._database.update_ddl([table_ddl, index_ddl])
        operation.result(timeout=_VECTOR_INDEX_DDL_TIMEOUT_SECONDS)
        logger.info(
            "Vector table and index %s created (or already existed).", self.index_name
        )

    def _create_table_if_not_exists(self) -> None:
        """Alias to create_index() for internal use."""
        self.create_index()

    def load_documents(self, documents: list[VectorStoreDocument]) -> None:
        """Load (upsert) documents into the Spanner vector table.

        Truncates the table before inserting (overwrite semantics) to remove
        stale documents from previous runs.
        """
        if not documents:
            return

        columns = (
            self.id_field,
            self.vector_field,
            self.DATA_FIELD,
            self.create_date_field,
            self.update_date_field,
        )
        rows = []
        for doc in documents:
            self._prepare_document(doc)
            if doc.vector is None:
                continue
            rows.append((
                str(doc.id),
                doc.vector,
                json.dumps(doc.data) if doc.data else None,
                doc.create_date,
                doc.update_date,
            ))

        safe_table = _safe_identifier(self.index_name)

        def _do_insert() -> None:
            with self._database.batch() as batch:
                batch.insert_or_update(
                    table=self.index_name, columns=columns, values=rows
                )

        def _is_table_not_found(exc: Exception) -> bool:
            return "Table not found" in str(exc)

        # Overwrite: truncate then insert.
        try:
            self._database.execute_partitioned_dml(
                f"DELETE FROM {safe_table} WHERE true"
            )
        except (exceptions.NotFound, exceptions.InvalidArgument) as e:
            if not _is_table_not_found(e):
                raise
            logger.info(
                "Table %s not found on load; creating it.", self.index_name
            )
            self._create_table_if_not_exists()
            _do_insert()
            return

        _do_insert()

    def _build_result(
        self,
        row: Any,
        select: list[str] | None,
        include_vectors: bool,
    ) -> VectorStoreSearchResult:
        """Build a VectorStoreSearchResult from a Spanner row.

        Row layout: id, vector, data JSON, create_date, update_date, distance
        """
        doc_id = row[0]
        vector = row[1] if include_vectors else None
        data_json = row[2]
        create_date = row[3]
        update_date = row[4]
        distance = row[5]

        data: dict[str, Any] = data_json if isinstance(data_json, dict) else {}
        if select:
            data = {k: v for k, v in data.items() if k in select}

        return VectorStoreSearchResult(
            document=VectorStoreDocument(
                id=doc_id,
                vector=vector,
                data=data,
                create_date=create_date,
                update_date=update_date,
            ),
            score=1.0 - distance,
        )

    def similarity_search_by_vector(
        self,
        query_embedding: list[float],
        k: int = 10,
        select: list[str] | None = None,
        filters: FilterExpr | None = None,
        include_vectors: bool = True,
    ) -> list[VectorStoreSearchResult]:
        """Perform ANN search by vector using Spanner COSINE_DISTANCE."""
        _t = _safe_identifier(self.index_name)
        _id = _safe_identifier(self.id_field)
        _vec = _safe_identifier(self.vector_field)
        _data = _safe_identifier(self.DATA_FIELD)
        _cdate = _safe_identifier(self.create_date_field)
        _udate = _safe_identifier(self.update_date_field)

        params: dict[str, Any] = {"query_vector": query_embedding, "k": k}
        param_types_map = {
            "query_vector": param_types.Array(param_types.FLOAT64),
            "k": param_types.INT64,
        }

        sql = (
            f"SELECT {_id}, {_vec}, {_data}, {_cdate}, {_udate},\n"
            f"       COSINE_DISTANCE({_vec}, @query_vector) AS distance\n"
            f"FROM {_t}\n"
            f"ORDER BY distance\n"
            f"LIMIT @k"
        )

        results: list[VectorStoreSearchResult] = []
        with self._database.snapshot() as snapshot:
            rows = snapshot.execute_sql(
                sql, params=params, param_types=param_types_map
            )
            for row in rows:
                result = self._build_result(row, select, include_vectors)
                if filters is None or filters.evaluate(result.document):
                    results.append(result)

        return results

    def search_by_id(
        self,
        id: str,
        select: list[str] | None = None,
        include_vectors: bool = True,
    ) -> VectorStoreDocument:
        """Search for a document by id."""
        _t = _safe_identifier(self.index_name)
        _id = _safe_identifier(self.id_field)
        _vec = _safe_identifier(self.vector_field)
        _data = _safe_identifier(self.DATA_FIELD)
        _cdate = _safe_identifier(self.create_date_field)
        _udate = _safe_identifier(self.update_date_field)

        id_param_type = (
            param_types.INT64 if isinstance(id, int) else param_types.STRING
        )
        sql = (
            f"SELECT {_id}, {_vec}, {_data}, {_cdate}, {_udate}\n"
            f"FROM {_t}\n"
            f"WHERE {_id} = @id"
        )

        with self._database.snapshot() as snapshot:
            rows = list(
                snapshot.execute_sql(
                    sql, params={"id": id}, param_types={"id": id_param_type}
                )
            )
            if rows:
                row = rows[0]
                doc_id = row[0]
                vector = row[1] if include_vectors else None
                data_json = row[2]
                create_date = row[3]
                update_date = row[4]

                data: dict[str, Any] = (
                    data_json if isinstance(data_json, dict) else {}
                )
                if select:
                    data = {k: v for k, v in data.items() if k in select}

                return VectorStoreDocument(
                    id=doc_id,
                    vector=vector,
                    data=data,
                    create_date=create_date,
                    update_date=update_date,
                )

        return VectorStoreDocument(id=id, vector=None, data={})

    def count(self) -> int:
        """Return the total number of documents in the store."""
        _t = _safe_identifier(self.index_name)
        try:
            with self._database.snapshot() as snapshot:
                rows = list(
                    snapshot.execute_sql(f"SELECT COUNT(*) FROM {_t}")
                )
                return int(rows[0][0]) if rows else 0
        except Exception:
            logger.warning("Error counting documents in Spanner table %s", self.index_name)
            return 0

    def remove(self, ids: list[str]) -> None:
        """Remove documents by id."""
        if not ids:
            return
        with self._database.batch() as batch:
            batch.delete(
                self.index_name,
                keyset=spanner.KeySet(keys=[[id_] for id_ in ids]),
            )

    def update(self, document: VectorStoreDocument) -> None:
        """Update a document in the store."""
        self._prepare_update(document)
        columns = (
            self.id_field,
            self.vector_field,
            self.DATA_FIELD,
            self.create_date_field,
            self.update_date_field,
        )
        with self._database.batch() as batch:
            batch.insert_or_update(
                table=self.index_name,
                columns=columns,
                values=[(
                    str(document.id),
                    document.vector,
                    json.dumps(document.data) if document.data else None,
                    document.create_date,
                    document.update_date,
                )],
            )

    def close(self) -> None:
        """Release the Spanner database resource."""
        if hasattr(self, "_database") and self._database:
            from graphrag_storage.spanner_resource_manager import (
                SpannerResourceManager,  # type: ignore[import]
            )

            logger.debug(
                "Releasing SpannerVectorStore database: obj_id=%s", id(self)
            )
            SpannerResourceManager.release_database(self._database)
            self._database = None  # type: ignore[assignment]
