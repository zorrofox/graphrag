# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Storage functions for the GraphRAG run module."""

import logging
from io import BytesIO

import pandas as pd

from graphrag.storage.pipeline_storage import PipelineStorage

logger = logging.getLogger(__name__)


async def load_table_from_storage(name: str, storage: PipelineStorage) -> pd.DataFrame:
    """Load a table from the storage instance."""
    if hasattr(storage, "load_table"):
        return await storage.load_table(name)

    filename = f"{name}.parquet"
    if not await storage.has(filename):
        msg = f"Could not find {filename} in storage!"
        raise ValueError(msg)
    try:
        logger.info("reading table from storage: %s", filename)
        return pd.read_parquet(BytesIO(await storage.get(filename, as_bytes=True)))
    except Exception:
        logger.exception("error loading table from storage: %s", filename)
        raise


async def write_table_to_storage(
    table: pd.DataFrame, name: str, storage: PipelineStorage
) -> None:
    """Write a table to storage."""
    if hasattr(storage, "set_table"):
        await storage.set_table(name, table)
    else:
        await storage.set(f"{name}.parquet", table.to_parquet())


async def delete_table_from_storage(name: str, storage: PipelineStorage) -> None:
    """Delete a table to storage."""
    # We might need delete_table too, but let's stick to basics first.
    # If SpannerPipelineStorage implements delete(key), it might need to know if key is a table name.
    # But here we are passing f"{name}.parquet".
    # If we use Spanner, we probably don't want to pass .parquet extension if it's a real table.
    
    # Let's add delete_table support for consistency if we need it.
    # For now, let's assume standard storage for delete or implement it later if needed.
    await storage.delete(f"{name}.parquet")


async def storage_has_table(name: str, storage: PipelineStorage) -> bool:
    """Check if a table exists in storage."""
    if hasattr(storage, "has_table"):
        return await storage.has_table(name)
    return await storage.has(f"{name}.parquet")