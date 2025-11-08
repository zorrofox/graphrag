# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Google Cloud Spanner implementation of PipelineStorage."""

import base64
import binascii
import json
import logging
import re
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from google.api_core import exceptions
from google.cloud import spanner

from graphrag.storage.pipeline_storage import PipelineStorage

logger = logging.getLogger(__name__)


class SpannerPipelineStorage(PipelineStorage):
    """The Google Cloud Spanner implementation."""

    def __init__(self, **kwargs: Any) -> None:
        project_id = kwargs.get("project_id")
        instance_id = kwargs.get("instance_id")
        database_id = kwargs.get("database_id")
        credentials = kwargs.get("credentials")
        
        # Fix: Ensure table_prefix is an empty string if it's None (default value from config)
        self._table_prefix = kwargs.get("table_prefix") or ""
        self._blob_table = f"{self._table_prefix}Blobs"

        if not all([project_id, instance_id, database_id]):
            msg = "project_id, instance_id, and database_id are required."
            raise ValueError(msg)

        self._client = spanner.Client(project=project_id, credentials=credentials)
        self._instance = self._client.instance(instance_id)
        self._database = self._instance.database(database_id)
        self._schema_cache = {}

    def _get_table_schema(self, table_name: str) -> dict[str, str]:
        """Get table schema from cache or Spanner."""
        if table_name in self._schema_cache:
            return self._schema_cache[table_name]
        
        try:
            with self._database.snapshot() as snapshot:
                results = snapshot.execute_sql(
                    "SELECT COLUMN_NAME, SPANNER_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = @table_name",
                    params={"table_name": table_name},
                    param_types={"table_name": spanner.param_types.STRING}
                )
                schema = {row[0]: row[1] for row in results}
                if schema:
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
        
        # Object types
        non_null = series.dropna()
        if len(non_null) == 0:
            return "STRING(MAX)"
        
        first_val = non_null.iloc[0]
        if isinstance(first_val, str):
            return "STRING(MAX)"
        if isinstance(first_val, bool):
            return "BOOL"
        if isinstance(first_val, (int, np.integer)):
            return "INT64"
        if isinstance(first_val, (float, np.floating)):
            return "FLOAT64"

        if isinstance(first_val, (list, tuple, np.ndarray)):
            found_non_empty = False
            for val in non_null:
                if len(val) > 0:
                    found_non_empty = True
                    first_elem = val[0]
                    if isinstance(first_elem, str):
                        return "ARRAY<STRING(MAX)>"
                    if isinstance(first_elem, (int, np.integer)):
                        return "ARRAY<INT64>"
                    if isinstance(first_elem, (float, np.floating)):
                        return "ARRAY<FLOAT64>"
                    if isinstance(first_elem, (dict, list, tuple)):
                        return "JSON"
                    # If mixed or unknown primitive, fallback to JSON
                    return "JSON"
            
            if not found_non_empty:
                 # All lists are empty. Default to ARRAY<STRING(MAX)> as it's most common for IDs in GraphRAG.
                 return "ARRAY<STRING(MAX)>"
            return "JSON"

        if isinstance(first_val, dict):
            return "JSON"
            
        return "STRING(MAX)"

    def _create_table_if_not_exists(self, table_name: str, df: pd.DataFrame) -> dict[str, str]:
        """Create a Spanner table based on DataFrame schema and return the inferred schema."""
        columns_ddl = []
        primary_key = "id" if "id" in df.columns else None
        inferred_schema = {}
        
        for col_name in df.columns:
            spanner_type = self._infer_spanner_type(df[col_name])
            inferred_schema[col_name] = spanner_type
            # Spanner PK must be NOT NULL
            nullable = " NOT NULL" if col_name == primary_key else ""
            columns_ddl.append(f"`{col_name}` {spanner_type}{nullable}")

        if not primary_key:
             # Fallback: use first column as PK if no 'id'
             primary_key = df.columns[0]
             # Ensure PK is NOT NULL in DDL, and update schema if needed (though type shouldn't change)
             columns_ddl[0] = f"`{primary_key}` {inferred_schema[primary_key]} NOT NULL"

        ddl = f"""CREATE TABLE `{table_name}` (
            {', '.join(columns_ddl)}
        ) PRIMARY KEY (`{primary_key}`)"""
        
        logger.info("Creating table %s with DDL: %s", table_name, ddl)
        
        operation = self._database.update_ddl([ddl])
        operation.result(timeout=600)
        logger.info("Table %s created successfully.", table_name)
        return inferred_schema

    def _alter_table_add_columns(self, table_name: str, df: pd.DataFrame) -> None:
        """Alter a Spanner table to add missing columns."""
        # We can use _get_table_schema here instead of raw query, but we want fresh data.
        # Let's force refresh by not using cache here, or just use the raw query as before.
        # Using raw query to be safe and sure we have latest state from Spanner.
        with self._database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = @table_name",
                params={"table_name": table_name},
                param_types={"table_name": spanner.param_types.STRING}
            )
            existing_columns = set(row[0] for row in results)

        new_columns = [col for col in df.columns if col not in existing_columns]
        
        if not new_columns:
            logger.warning("Column not found error received, but no new columns detected for table %s.", table_name)
            return

        alter_statements = []
        for col in new_columns:
            spanner_type = self._infer_spanner_type(df[col])
            alter_statements.append(f"ALTER TABLE `{table_name}` ADD COLUMN `{col}` {spanner_type}")
        
        logger.info("Altering table %s with DDL: %s", table_name, alter_statements)
        
        operation = self._database.update_ddl(alter_statements)
        operation.result(timeout=600)
        logger.info("Table %s altered successfully.", table_name)

    def _batch_insert(self, table_name: str, columns: tuple[str, ...], values: list[tuple[Any, ...]]) -> None:
        """Helper to perform batch inserts."""
        chunk_size = 500
        for i in range(0, len(values), chunk_size):
            chunk = values[i : i + chunk_size]
            with self._database.batch() as batch:
                batch.insert_or_update(
                    table=table_name,
                    columns=columns,
                    values=chunk,
                )

    def _prepare_values(self, df: pd.DataFrame, columns: tuple[str, ...], schema: dict[str, str]) -> list[tuple[Any, ...]]:
        """Prepare DataFrame values for Spanner insertion, handling JSON serialization based on schema."""
        values = []
        for _, row in df.iterrows():
            row_values = []
            for col in columns:
                val = row[col]
                
                # Handle numpy arrays (convert to list)
                if isinstance(val, np.ndarray):
                    val = val.tolist()

                col_type = schema.get(col, "").upper()
                
                if "JSON" in col_type:
                     # FORCE JSON dump for list/dict if column is JSON
                     if isinstance(val, (dict, list)):
                         val = json.dumps(val)
                elif isinstance(val, (dict, list)):
                     # Fallback heuristic when schema is unknown
                     if isinstance(val, dict) or (isinstance(val, list) and len(val) > 0 and isinstance(val[0], (dict, list))):
                         val = json.dumps(val)
                
                row_values.append(val)
            values.append(tuple(row_values))
        return values

    async def set_table(self, name: str, table: pd.DataFrame) -> None:
        """Write a dataframe to a Spanner table, creating or altering it if necessary."""
        table_name = f"{self._table_prefix}{name}"
        df = table.where(pd.notnull(table), None)
        columns = tuple(df.columns)

        max_retries = 2
        for attempt in range(max_retries + 1):
            # Get schema (from cache or fresh if not in cache)
            schema = self._get_table_schema(table_name)
            values = self._prepare_values(df, columns, schema)

            try:
                self._batch_insert(table_name, columns, values)
                return # Success
            except exceptions.NotFound as e:
                if attempt >= max_retries:
                    raise
                
                msg = str(e)
                if "Table not found" in msg:
                     logger.info("Table %s not found, attempting to create it.", table_name)
                     new_schema = self._create_table_if_not_exists(table_name, df)
                     self._schema_cache[table_name] = new_schema
                elif "Column not found" in msg:
                     logger.info("Column mismatch for table %s, attempting to alter it.", table_name)
                     self._alter_table_add_columns(table_name, df)
                     self._schema_cache.pop(table_name, None) # Clear cache
                else:
                    raise
            except exceptions.FailedPrecondition as e:
                 # Handle "Expected JSON" error if it happens despite our best efforts,
                 # maybe schema cache was stale.
                 if "Expected JSON" in str(e) and attempt < max_retries:
                      logger.warning("Got Expected JSON error for table %s, refreshing schema and retrying.", table_name)
                      self._schema_cache.pop(table_name, None)
                      continue
                 raise

    async def load_table(self, name: str) -> pd.DataFrame:
        """Load a table from Spanner."""
        table_name = f"{self._table_prefix}{name}"
        with self._database.snapshot() as snapshot:
            results = snapshot.execute_sql(f"SELECT * FROM {table_name}")
            rows = list(results)
            
            columns = []
            if results.fields:
                columns = [field.name for field in results.fields]
            else:
                # Fallback: query INFORMATION_SCHEMA if fields are missing
                try:
                    schema_results = snapshot.execute_sql(
                        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = @table_name ORDER BY ORDINAL_POSITION",
                        params={"table_name": table_name},
                        param_types={"table_name": spanner.param_types.STRING}
                    )
                    columns = [row[0] for row in schema_results]
                except Exception as e:
                    logger.warning("Failed to fetch schema for table %s: %s", table_name, e)

            return pd.DataFrame(rows, columns=columns if columns else None)

    async def has_table(self, name: str) -> bool:
        """Check if a table exists."""
        table_name = f"{self._table_prefix}{name}"
        try:
            with self._database.snapshot() as snapshot:
                # Use a cheap query to check existence
                list(snapshot.execute_sql(f"SELECT 1 FROM {table_name} LIMIT 1"))
            return True
        except Exception:
            return False

    async def get(
        self, key: str, as_bytes: bool | None = False, encoding: str | None = None
    ) -> Any:
        """Get a blob from Spanner."""
        try:
            with self._database.snapshot() as snapshot:
                # Using read instead of execute_sql for simple key lookup
                result = snapshot.read(
                    self._blob_table, ["value"], keyset=spanner.KeySet(keys=[[key]])
                )
                row = None
                for r in result:
                    row = r
                    break
                
                if not row:
                    return None
                data = row[0]
                
                # Workaround: If data is bytes and looks like base64, try to decode it.
                if isinstance(data, bytes):
                    try:
                        decoded = base64.b64decode(data, validate=True)
                        # Stronger validation: re-encode must match exactly.
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

    async def set(self, key: str, value: Any, encoding: str | None = None) -> None:
        """Set a blob in Spanner."""
        if isinstance(value, str):
            value = value.encode(encoding or "utf-8")

        if not isinstance(value, bytes):
             try:
                 value = str(value).encode(encoding or "utf-8")
             except Exception:
                 raise ValueError(f"Spanner blob storage expects bytes, got {type(value)}")

        # Use FROM_BASE64 in DML to bypass client-side type inference issues.
        value_base64 = base64.b64encode(value).decode("ascii")

        def _write_blob(transaction):
            # DEBUG: Embed base64 directly in SQL to rule out parameter binding issues.
            # WARNING: This is vulnerable to SQL injection if value_base64 wasn't strictly base64.
            # Since it comes from base64.b64encode, it should be safe-ish for debugging.
            sql = f"INSERT OR UPDATE {self._blob_table} (key, value, updated_at) VALUES (@key, FROM_BASE64('{value_base64}'), PENDING_COMMIT_TIMESTAMP())"
            transaction.execute_update(
                sql,
                params={"key": key},
                param_types={"key": spanner.param_types.STRING},
            )

        try:
            self._database.run_in_transaction(_write_blob)
        except exceptions.NotFound as e:
             if "Table not found" in str(e) and self._blob_table in str(e):
                 logger.info("Blob table %s not found, attempting to create it.", self._blob_table)
                 # Create Blobs table specifically
                 ddl = f"""CREATE TABLE `{self._blob_table}` (
                        key STRING(MAX) NOT NULL,
                        value BYTES(MAX),
                        updated_at TIMESTAMP OPTIONS (allow_commit_timestamp=true),
                    ) PRIMARY KEY (key)"""
                 operation = self._database.update_ddl([ddl])
                 operation.result(timeout=600)
                 # Retry write
                 self._database.run_in_transaction(_write_blob)
             else:
                 raise

    async def has(self, key: str) -> bool:
        """Check if a blob exists in Spanner."""
        try:
            with self._database.snapshot() as snapshot:
                result = snapshot.read(
                    self._blob_table, ["key"], keyset=spanner.KeySet(keys=[[key]])
                )
                for _ in result:
                    return True
            return False
        except Exception:
            return False

    async def delete(self, key: str) -> None:
        """Delete a blob from Spanner."""
        with self._database.batch() as batch:
            batch.delete(self._blob_table, keyset=spanner.KeySet(keys=[[key]]))

    async def clear(self) -> None:
        """Clear all blobs."""
        try:
            self._database.execute_partitioned_dml(f"DELETE FROM {self._blob_table} WHERE true")
        except Exception:
             logger.exception("Error clearing Spanner blobs")
             raise

    def child(self, name: str | None) -> "PipelineStorage":
        """Create a child storage instance."""
        if name is None:
            return self
        # For simplicity, I'll just raise NotImplemented for now.
        raise NotImplementedError("Child storage not implemented for Spanner yet.")

    def keys(self) -> list[str]:
        """List all blob keys."""
        with self._database.snapshot() as snapshot:
            result = snapshot.execute_sql(f"SELECT key FROM {self._blob_table}")
            return [row[0] for row in result]

    async def get_creation_date(self, key: str) -> str:
        """Get creation date (updated_at) for a blob."""
        with self._database.snapshot() as snapshot:
            result = snapshot.read(
                self._blob_table, ["updated_at"], keyset=spanner.KeySet(keys=[[key]])
            )
            for row in result:
                if row[0]:
                    return str(row[0]) # Simplistic formatting
            return ""

    def find(
        self,
        file_pattern: re.Pattern[str],
        base_dir: str | None = None,
        file_filter: dict[str, Any] | None = None,
        max_count=-1,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """Find blobs."""
        all_keys = self.keys()
        num_loaded = 0
        for key in all_keys:
            match = file_pattern.search(key)
            if match:
                if base_dir and not key.startswith(base_dir):
                    continue
                yield (key, match.groupdict())
                num_loaded += 1
                if max_count > 0 and num_loaded >= max_count:
                    break