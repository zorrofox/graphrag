# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""Cloud Run Job entrypoint for GraphRAG indexing and incremental update."""

import asyncio
import logging
import os
import sys
from pathlib import Path

import graphrag.api as api
from graphrag.config.enums import IndexingMethod
from graphrag.config.load_config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

CONFIG_DIR = Path(os.getenv("GRAPHRAG_CONFIG_DIR", "/app/config"))
IS_UPDATE = os.getenv("GRAPHRAG_IS_UPDATE", "false").lower() == "true"


async def run() -> None:
    """Run the indexing or update pipeline."""
    mode = "update" if IS_UPDATE else "index"
    logger.info("Starting GraphRAG %s pipeline | config_dir=%s", mode, CONFIG_DIR)

    config = load_config(root_dir=CONFIG_DIR)

    results = await api.build_index(
        config=config,
        method=IndexingMethod.Standard,
        is_update_run=IS_UPDATE,
        verbose=True,
    )

    errors = [r for r in results if r.error]
    if errors:
        for result in errors:
            logger.error("Workflow %s failed: %s", result.workflow, result.error)
        sys.exit(1)

    logger.info(
        "GraphRAG %s pipeline completed successfully (%d workflows)",
        mode,
        len(results),
    )


if __name__ == "__main__":
    asyncio.run(run())
