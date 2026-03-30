# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Startup loader: reads parquet index files from GCS into memory."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from graphrag_storage import create_storage
from graphrag_storage.tables.table_provider_factory import create_table_provider

from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.data_reader import DataReader

logger = logging.getLogger(__name__)


@dataclass
class IndexData:
    """In-memory snapshot of the GraphRAG index, loaded from GCS parquet files."""

    config: GraphRagConfig
    entities: pd.DataFrame
    communities: pd.DataFrame
    community_reports: pd.DataFrame
    text_units: pd.DataFrame
    relationships: pd.DataFrame
    covariates: pd.DataFrame | None


async def load_index(config_dir: Path) -> IndexData:
    """Load all required DataFrames from GCS at service startup.

    All search types (global/local/drift/basic) share this single
    preloaded snapshot. The Spanner vector store is accessed per-request
    via the config (not preloaded) since it maintains its own connection pool.
    """
    logger.info("Loading GraphRAG config from %s", config_dir)
    config = load_config(root_dir=config_dir)

    storage = create_storage(config.output_storage)
    table_provider = create_table_provider(config.table_provider, storage=storage)
    reader = DataReader(table_provider)

    logger.info("Loading index DataFrames from GCS (output_storage: %s)", config.output_storage.bucket_name)

    entities, communities, community_reports, text_units, relationships = await asyncio.gather(
        reader.entities(),
        reader.communities(),
        reader.community_reports(),
        reader.text_units(),
        reader.relationships(),
    )

    covariates: pd.DataFrame | None = None
    if await table_provider.has("covariates"):
        covariates = await reader.covariates()
        logger.info("Loaded optional covariates table")

    logger.info(
        "Index loaded — entities=%d communities=%d reports=%d text_units=%d relationships=%d covariates=%s",
        len(entities),
        len(communities),
        len(community_reports),
        len(text_units),
        len(relationships),
        len(covariates) if covariates is not None else "N/A",
    )

    return IndexData(
        config=config,
        entities=entities,
        communities=communities,
        community_reports=community_reports,
        text_units=text_units,
        relationships=relationships,
        covariates=covariates,
    )
