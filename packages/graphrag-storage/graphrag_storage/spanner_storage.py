# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Google Cloud Spanner implementation of Storage."""

import asyncio
import base64
import binascii
import json
import logging
import re
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from google.api_core import exceptions
from google.cloud import spanner

from graphrag_storage.spanner_resource_manager import SpannerResourceManager
from graphrag_storage.storage import (
    Storage,
    get_timestamp_formatted_with_local_tz,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL timeout
# ---------------------------------------------------------------------------

_DDL_TIMEOUT_SECONDS: int = 600

# ---------------------------------------------------------------------------
# Table-name sanitisation
# ---------------------------------------------------------------------------

_UNSAFE_TABLE_CHARS_RE = re.compile(r"[^A-Za-z0-9_]")

# ---------------------------------------------------------------------------
# SQL-injection guard
# ---------------------------------------------------------------------------

_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# ---------------------------------------------------------------------------
# Schema cache size limit
# ---------------------------------------------------------------------------

_SCHEMA_CACHE_MAX_SIZE = 256


def _safe_identifier(name: str) -> str:
    """Return a backtick-quoted Spanner identifier after strict validation."""
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Unsafe Spanner identifier {name!r}: only letters, digits, and "
            "underscores are permitted, and the name must not start with a digit."
        )
    return f"`{name}`"


class SpannerStorage(Storage):
    """The Google Cloud Spanner implementation of Storage."""

    def _sanitize_table_name(self, name: str) -> str:
        """Sanitize *name* so it is a valid Spanner identifier."""
        sanitized = _UNSAFE_TABLE_CHARS_RE.sub("_", name)
        if sanitized and sanitized[0].isdigit():
            sanitized = f"t_{sanitized}"
        return sanitized

    def __init__(self, **kwargs: Any) -> None:
        project_id = kwargs.get("project_id")
        instance_id = kwargs.get("instance_id")
        database_id = kwargs.get("database_id")
        credentials = kwargs.get("credentials")

        raw_prefix = kwargs.get("table_prefix") or ""
        self._table_prefix = self._sanitize_table_name(raw_prefix)
        self._blob_table = f"{self._table_prefix}Blobs"

        logger.debug(
            "Creating SpannerStorage: obj_id=%s, prefix=%s", id(self), self._table_prefix
        )

        if not all([project_id, instance_id, database_id]):
            msg = "project_id, instance_id, and database_id are required."
            raise ValueError(msg)

        self._project_id: str = project_id  # type: ignore[assignment]
        self._instance_id: str = instance_id  # type: ignore[assignment]
        self._database_id: str = database_id  # type: ignore[assignment]
        self._credentials = credentials

        self._database: Any = SpannerResourceManager.get_database(
            project_id=self._project_id,
            instance_id=self._instance_id,
            database_id=self._database_id,
            credentials=credentials,
        )
        self._schema_cache: OrderedDict[str, dict[str, str]] = OrderedDict()

    def _get_table_schema(self, table_name: str) -> dict[str, str]:
        """Get table schema from cache or Spanner."""
        if table_name in self._schema_cache:
            return self._schema_cache[table_name]

        try:
            with self._database.snapshot() as snapshot:
                results = snapshot.execute_sql(
                    "SELECT COLUMN_NAME, SPANNER_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = @table_name",
                    params={"table_name": table_name},
                    param_types={"table_name": spanner.param_types.STRING},
                )
                schema = {row[0]: row[1] for row in results}
                if schema:
                    if len(self._schema_cache) >= _SCHEMA_CACHE_MAX_SIZE:
                        self._schema_cache.popitem(last=False)
                    self._schema_cache[table_name] = schema
                return schema
        except Exception as e:
            logger.warning("Failed to get schema for table %s: %s", table_name, e)
            return {}

    def _infer_spanner_type(self, series: pd.Series) -> str:
        """Infer Spanner data type from a pandas Series."""
        if pd.api.types.is_integer_dtype(series):
            return "INT64"
        if pd.api.types.is_float_dtype(series):
            return "FLOAT64"
        if pd.api.types.is_bool_dtype(series):
            return "BOOL"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "TIMESTAMP"

        non_null = series.dropna()
        if len(non_null) == 0:
            return "STRING(MAX)"

        SAMPLE_SIZE = 10
        samples = non_null.head(SAMPLE_SIZE)

        detected: list[str] = []
        for val in samples:
            if isinstance(val, bool):
                detected.append("BOOL")
            elif isinstance(val, str):
                detected.append("STRING(MAX)")
            elif isinstance(val, (int, np.integer)):
                detected.append("INT64")
            elif isinstance(val, (float, np.floating)):
                detected.append("FLOAT64")
            elif isinstance(val, dict):
                detected.append("JSON")
            elif isinstance(val, (list, tuple, np.ndarray)):
                if len(val) == 0:
                    detected.append("ARRAY<STRING(MAX)>")
                else:
                    first_elem = val[0]
                    if isinstance(first_elem, str):
                        detected.append("ARRAY<STRING(MAX)>")
                    elif isinstance(first_elem, (int, np.integer)):
                        detected.append("ARRAY<INT64>")
                    elif isinstance(first_elem, (float, np.floating)):
                        detected.append("ARRAY<FLOAT64>")
                    elif isinstance(first_elem, (dict, list, tuple)):
                        detected.append("JSON")
                    else:
                        detected.append("JSON")
            else:
                detected.append("STRING(MAX)")

        if not detected:
            return "STRING(MAX)"

        most_common = max(set(detected), key=detected.count)
        if detected.count(most_common) < len(detected):
            return "JSON"
        return most_common

    def _create_table_if_not_exists(
        self, table_name: str, df: pd.DataFrame
    ) -> dict[str, str]:
        """Create a Spanner table based on DataFrame schema."""
        columns_ddl = []
        primary_key = "id" if "id" in df.columns else None
        inferred_schema = {}

        for col_name in df.columns:
            spanner_type = self._infer_spanner_type(df[col_name])
            inferred_schema[col_name] = spanner_type
            nullable = " NOT NULL" if col_name == primary_key else ""
            columns_ddl.append(f"`{col_name}` {spanner_type}{nullable}")

        if not primary_key:
            primary_key = df.columns[0]
            columns_ddl[0] = (
                f"`{primary_key}` {inferred_schema[primary_key]} NOT NULL"
            )

        ddl = (
            f"CREATE TABLE `{table_name}` (\n"
            f"    {', '.join(columns_ddl)}\n"
            f") PRIMARY KEY (`{primary_key}`)"
        )
        logger.info("Creating table %s with DDL: %s", table_name, ddl)
        operation = self._database.update_ddl([ddl])
        operation.result(timeout=_DDL_TIMEOUT_SECONDS)
        logger.info("Table %s created successfully.", table_name)
        return inferred_schema

    def _alter_table_add_columns(
        self, table_name: str, df: pd.DataFrame
    ) -> None:
        """Alter a Spanner table to add missing columns."""
        with self._database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = @table_name",
                params={"table_name": table_name},
                param_types={"table_name": spanner.param_types.STRING},
            )
            existing_columns = {row[0] for row in results}

        new_columns = [col for col in df.columns if col not in existing_columns]
        if not new_columns:
            logger.warning(
                "Column not found error received, but no new columns detected for table %s.",
                table_name,
            )
            return

        alter_statements = [
            f"ALTER TABLE `{table_name}` ADD COLUMN `{col}` {self._infer_spanner_type(df[col])}"
            for col in new_columns
        ]
        logger.info("Altering table %s with DDL: %s", table_name, alter_statements)
        operation = self._database.update_ddl(alter_statements)
        operation.result(timeout=_DDL_TIMEOUT_SECONDS)
        logger.info("Table %s altered successfully.", table_name)

    def _batch_insert(
        self,
        table_name: str,
        columns: tuple[str, ...],
        values: list[tuple[Any, ...]],
    ) -> None:
        """Perform batch inserts in chunks."""
        chunk_size = 500
        for i in range(0, len(values), chunk_size):
            chunk = values[i : i + chunk_size]
            with self._database.batch() as batch:
                batch.insert_or_update(
                    table=table_name,
                    columns=columns,
                    values=chunk,
                )

    def _prepare_values(
        self,
        df: pd.DataFrame,
        columns: tuple[str, ...],
        schema: dict[str, str],
    ) -> list[tuple[Any, ...]]:
        """Prepare DataFrame values for Spanner insertion."""
        col_data: list[list[Any]] = []
        for col in columns:
            raw: list[Any] = df[col].tolist()
            col_type = schema.get(col, "").upper()
            processed: list[Any] = []
            if "JSON" in col_type:
                for v in raw:
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    processed.append(
                        json.dumps(v) if isinstance(v, (dict, list)) else v
                    )
            else:
                for v in raw:
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    if isinstance(v, dict) or (
                        isinstance(v, list)
                        and v
                        and isinstance(v[0], (dict, list))
                    ):
                        v = json.dumps(v)
                    processed.append(v)
            col_data.append(processed)
        return list(zip(*col_data))

    async def set_table(self, name: str, table: pd.DataFrame) -> None:
        """Write a DataFrame to a Spanner table, auto-creating/altering schema as needed."""
        sanitized_name = self._sanitize_table_name(name)
        table_name = f"{self._table_prefix}{sanitized_name}"
        df = table.where(pd.notnull(table), other=None)  # type: ignore[arg-type]
        columns = tuple(df.columns)

        max_retries = 2
        for attempt in range(max_retries + 1):
            schema = self._get_table_schema(table_name)
            values = self._prepare_values(df, columns, schema)

            try:
                await asyncio.to_thread(self._batch_insert, table_name, columns, values)
                return
            except exceptions.NotFound as e:
                if attempt >= max_retries:
                    raise
                msg = str(e)
                if "Table not found" in msg:
                    logger.info(
                        "Table %s not found, attempting to create it.", table_name
                    )
                    new_schema = await asyncio.to_thread(
                        self._create_table_if_not_exists, table_name, df
                    )
                    self._schema_cache[table_name] = new_schema
                elif "Column not found" in msg:
                    logger.info(
                        "Column mismatch for table %s, attempting to alter it.",
                        table_name,
                    )
                    await asyncio.to_thread(
                        self._alter_table_add_columns, table_name, df
                    )
                    self._schema_cache.pop(table_name, None)
                else:
                    raise
            except exceptions.FailedPrecondition as e:
                if "Expected JSON" in str(e) and attempt < max_retries:
                    logger.warning(
                        "Got Expected JSON error for table %s, refreshing schema and retrying.",
                        table_name,
                    )
                    self._schema_cache.pop(table_name, None)
                    continue
                raise

    async def load_table(
        self, name: str, limit: int | None = None, offset: int = 0
    ) -> pd.DataFrame:
        """Load a table from Spanner with optional pagination."""
        return await asyncio.to_thread(self._load_table_sync, name, limit, offset)

    def _load_table_sync(
        self, name: str, limit: int | None = None, offset: int = 0
    ) -> pd.DataFrame:
        """Synchronous core for load_table()."""
        sanitized_name = self._sanitize_table_name(name)
        table_name = f"{self._table_prefix}{sanitized_name}"

        sql = f"SELECT * FROM {_safe_identifier(table_name)}"
        params: dict = {}
        param_types_map: dict = {}
        if limit is not None:
            sql += " LIMIT @lim OFFSET @off"
            params = {"lim": limit, "off": offset}
            param_types_map = {
                "lim": spanner.param_types.INT64,
                "off": spanner.param_types.INT64,
            }

        with self._database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                sql,
                params=params or None,
                param_types=param_types_map or None,
            )
            rows = list(results)

            columns = []
            if results.fields:
                columns = [field.name for field in results.fields]
            else:
                try:
                    schema_results = snapshot.execute_sql(
                        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                        "WHERE TABLE_NAME = @table_name ORDER BY ORDINAL_POSITION",
                        params={"table_name": table_name},
                        param_types={"table_name": spanner.param_types.STRING},
                    )
                    columns = [row[0] for row in schema_results]
                except Exception as e:
                    logger.warning(
                        "Failed to fetch schema for table %s: %s", table_name, e
                    )

            return pd.DataFrame(rows, columns=columns if columns else None)

    def _has_table_sync(self, name: str) -> bool:
        """Synchronous core for has_table()."""
        sanitized_name = self._sanitize_table_name(name)
        table_name = f"{self._table_prefix}{sanitized_name}"
        try:
            with self._database.snapshot() as snapshot:
                list(snapshot.execute_sql(
                    f"SELECT 1 FROM {_safe_identifier(table_name)} LIMIT 1"
                ))
            return True
        except Exception:
            return False

    async def has_table(self, name: str) -> bool:
        """Check if a named table exists in Spanner."""
        return await asyncio.to_thread(self._has_table_sync, name)

    # ------------------------------------------------------------------
    # Blob (key/value) operations — Storage ABC implementation
    # ------------------------------------------------------------------

    def _get_sync(
        self, key: str, as_bytes: bool | None = False, encoding: str | None = None
    ) -> Any:
        """Synchronous core for get()."""
        try:
            with self._database.snapshot() as snapshot:
                result = snapshot.read(
                    self._blob_table,
                    ["value"],
                    keyset=spanner.KeySet(keys=[[key]]),
                )
                row = None
                for r in result:
                    row = r
                    break

                if not row:
                    return None
                data = row[0]

                if isinstance(data, bytes):
                    try:
                        decoded = base64.b64decode(data, validate=True)
                        if base64.b64encode(decoded) == data:
                            data = decoded
                    except (binascii.Error, ValueError):
                        pass

                if as_bytes:
                    return data
                return data.decode(encoding or "utf-8")
        except Exception:
            logger.warning("Error getting key %s from Spanner", key)
            return None

    async def get(
        self, key: str, as_bytes: bool | None = False, encoding: str | None = None
    ) -> Any:
        """Get a blob from Spanner."""
        return await asyncio.to_thread(self._get_sync, key, as_bytes, encoding)

    def _ensure_blobs_table_exists(self) -> None:
        """Ensure the Blobs table exists."""
        logger.info("Ensuring Blob table %s exists.", self._blob_table)
        ddl = (
            f"CREATE TABLE IF NOT EXISTS `{self._blob_table}` (\n"
            "    `key`        STRING(MAX) NOT NULL,\n"
            "    `value`      BYTES(MAX),\n"
            "    `created_at` TIMESTAMP OPTIONS (allow_commit_timestamp=true),\n"
            "    `updated_at` TIMESTAMP OPTIONS (allow_commit_timestamp=true)\n"
            f") PRIMARY KEY (`key`)"
        )
        operation = self._database.update_ddl([ddl])
        operation.result(timeout=_DDL_TIMEOUT_SECONDS)

    async def set(self, key: str, value: Any, encoding: str | None = None) -> None:
        """Set a blob in Spanner."""
        if isinstance(value, str):
            value = value.encode(encoding or "utf-8")
        if not isinstance(value, bytes):
            try:
                value = str(value).encode(encoding or "utf-8")
            except Exception:
                raise ValueError(
                    f"Spanner blob storage expects bytes, got {type(value)}"
                )

        value_base64 = base64.b64encode(value).decode("ascii")
        safe_table = _safe_identifier(self._blob_table)

        def _try_insert(transaction) -> None:
            transaction.execute_update(
                f"INSERT INTO {safe_table} (key, value, created_at, updated_at) "
                "VALUES (@key, FROM_BASE64(@val), PENDING_COMMIT_TIMESTAMP(), PENDING_COMMIT_TIMESTAMP())",
                params={"key": key, "val": value_base64},
                param_types={
                    "key": spanner.param_types.STRING,
                    "val": spanner.param_types.STRING,
                },
            )

        def _do_update(transaction) -> None:
            transaction.execute_update(
                f"UPDATE {safe_table} SET value = FROM_BASE64(@val), "
                "updated_at = PENDING_COMMIT_TIMESTAMP() WHERE key = @key",
                params={"key": key, "val": value_base64},
                param_types={
                    "key": spanner.param_types.STRING,
                    "val": spanner.param_types.STRING,
                },
            )

        try:
            await asyncio.to_thread(self._database.run_in_transaction, _try_insert)
        except exceptions.AlreadyExists:
            await asyncio.to_thread(self._database.run_in_transaction, _do_update)
        except (exceptions.NotFound, exceptions.InvalidArgument) as e:
            if "Table not found" in str(e):
                logger.info("Table not found error caught in set(): %s", e)
                await asyncio.to_thread(self._ensure_blobs_table_exists)
                await asyncio.to_thread(self._database.run_in_transaction, _try_insert)
            else:
                raise

    def _has_sync(self, key: str) -> bool:
        """Synchronous core for has()."""
        try:
            with self._database.snapshot() as snapshot:
                result = snapshot.read(
                    self._blob_table,
                    ["key"],
                    keyset=spanner.KeySet(keys=[[key]]),
                )
                for _ in result:
                    return True
            return False
        except Exception:
            return False

    async def has(self, key: str) -> bool:
        """Check if a blob exists in Spanner."""
        return await asyncio.to_thread(self._has_sync, key)

    def _delete_sync(self, key: str) -> None:
        """Synchronous core for delete()."""
        with self._database.batch() as batch:
            batch.delete(self._blob_table, keyset=spanner.KeySet(keys=[[key]]))

    async def delete(self, key: str) -> None:
        """Delete a blob from Spanner."""
        await asyncio.to_thread(self._delete_sync, key)

    def _clear_sync(self) -> None:
        """Synchronous core for clear()."""
        try:
            self._database.execute_partitioned_dml(
                f"DELETE FROM {_safe_identifier(self._blob_table)} WHERE true"
            )
        except Exception:
            logger.exception("Error clearing Spanner blobs")
            raise

    async def clear(self) -> None:
        """Clear all blobs."""
        await asyncio.to_thread(self._clear_sync)

    def child(self, name: str | None) -> "Storage":
        """Create a child storage instance with a nested table prefix."""
        if name is None:
            return self
        sanitized = self._sanitize_table_name(name)
        new_prefix = f"{self._table_prefix}{sanitized}_"
        return SpannerStorage(
            project_id=self._project_id,
            instance_id=self._instance_id,
            database_id=self._database_id,
            credentials=self._credentials,
            table_prefix=new_prefix,
        )

    def keys(self) -> list[str]:
        """List all blob keys."""
        with self._database.snapshot() as snapshot:
            result = snapshot.execute_sql(
                f"SELECT key FROM {_safe_identifier(self._blob_table)}"
            )
            return [row[0] for row in result]

    def find(
        self,
        file_pattern: re.Pattern[str],
    ) -> Iterator[str]:
        """Find blobs matching *file_pattern*."""
        for key in self.keys():
            if file_pattern.search(key):
                yield key

    def _get_creation_date_sync(self, key: str) -> str:
        """Synchronous core for get_creation_date()."""
        with self._database.snapshot() as snapshot:
            result = snapshot.read(
                self._blob_table,
                ["created_at"],
                keyset=spanner.KeySet(keys=[[key]]),
            )
            for row in result:
                if row[0]:
                    return get_timestamp_formatted_with_local_tz(row[0])
            return ""

    async def get_creation_date(self, key: str) -> str:
        """Get creation date for a blob."""
        return await asyncio.to_thread(self._get_creation_date_sync, key)

    def close(self) -> None:
        """Release the Spanner database resource."""
        if hasattr(self, "_database") and self._database:
            logger.debug(
                "Releasing SpannerStorage database: obj_id=%s", id(self)
            )
            SpannerResourceManager.release_database(self._database)
            self._database = None  # type: ignore[assignment]
