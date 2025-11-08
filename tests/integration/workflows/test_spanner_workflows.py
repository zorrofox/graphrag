# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import os
import pytest
import pandas as pd
import numpy as np
from uuid import uuid4

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.enums import ModelType
import graphrag.config.defaults as defs
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.stats import PipelineRunStats
from graphrag.index.typing.state import PipelineState
from graphrag.storage.spanner_pipeline_storage import SpannerPipelineStorage
from graphrag.index.workflows.create_final_documents import run_workflow as run_create_final_documents
from graphrag.index.workflows.create_final_text_units import run_workflow as run_create_final_text_units
from graphrag.storage.memory_pipeline_storage import MemoryPipelineStorage
from graphrag.cache.memory_pipeline_cache import InMemoryCache
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks

# Only run if explicitly enabled
pytestmark = pytest.mark.skipif(
    not os.environ.get("GRAPHRAG_GCP_INTEGRATION_TEST"),
    reason="GCP integration tests not enabled",
)

@pytest.fixture
def spanner_config():
    return {
        "project_id": os.environ.get("GCP_PROJECT_ID"),
        "instance_id": os.environ.get("SPANNER_INSTANCE_ID"),
        "database_id": os.environ.get("SPANNER_DATABASE_ID"),
    }

@pytest.mark.asyncio
async def test_create_final_documents_on_spanner(spanner_config):
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    # 1. Setup Spanner Storage with unique prefix
    prefix = f"WFLTest_DOC_{uuid4().hex[:8]}_"
    storage = SpannerPipelineStorage(**spanner_config, table_prefix=prefix)

    # 2. Prepare initial data (simulating output from previous steps)
    # documents table (initially just has raw text info)
    documents_df = pd.DataFrame({
        "id": ["doc1", "doc2"],
        "title": ["Document 1", "Document 2"],
        "text": ["Full text of doc 1", "Full text of doc 2"],
        # Important: Add creation_date to verify it's preserved
        "creation_date": ["2024-01-01", "2024-01-02"]
    })
    
    # text_units table (chunks of documents)
    text_units_df = pd.DataFrame({
        "id": ["unit1", "unit2", "unit3"],
        "text": ["chunk1 of doc1", "chunk2 of doc1", "chunk1 of doc2"],
        "n_tokens": [10, 10, 10],
        "document_ids": [["doc1"], ["doc1"], ["doc2"]]
    })

    print(f"Creating initial tables with prefix: {prefix}")
    # Write initial tables to Spanner
    await storage.set_table("documents", documents_df)
    await storage.set_table("text_units", text_units_df)

    # 3. Setup Pipeline Context
    config = GraphRagConfig(
        models={
            defs.DEFAULT_CHAT_MODEL_ID: LanguageModelConfig(
                type=ModelType.Chat,
                model="mock-model",
                model_provider="openai",
                api_key="mock-api-key",
            ),
            defs.DEFAULT_EMBEDDING_MODEL_ID: LanguageModelConfig(
                type=ModelType.Embedding,
                model="mock-embedding-model",
                model_provider="openai",
                api_key="mock-api-key",
            ),
        }
    )
    stats = PipelineRunStats()
    context = PipelineRunContext(
        stats=stats,
        input_storage=MemoryPipelineStorage(), # Not used by this workflow
        output_storage=storage, # THIS is what we are testing
        cache=InMemoryCache(),
        previous_storage=MemoryPipelineStorage(),
        callbacks=NoopWorkflowCallbacks(),
        state=PipelineState(),
    )

    try:
        # 4. Run the workflow
        print("Running create_final_documents workflow...")
        # This will load 'documents' and 'text_units' from Spanner,
        # join them, and write updated 'documents' back to Spanner.
        await run_create_final_documents(config, context)
        print("Workflow finished.")

        # 5. Verify output in Spanner
        final_documents = await storage.load_table("documents")
        print("Loaded final documents from Spanner.")
        
        assert len(final_documents) == 2
        final_documents = final_documents.sort_values("id").reset_index(drop=True)

        # Check doc1
        assert final_documents.iloc[0]["id"] == "doc1"
        # Spanner might return list for ARRAY, need to handle potential ordering if not guaranteed,
        # but here we inserted them in order.
        # Actually, create_final_documents aggregates them, order might depend on text_units order.
        doc1_units = set(final_documents.iloc[0]["text_unit_ids"])
        assert doc1_units == {"unit1", "unit2"}
        assert final_documents.iloc[0]["creation_date"] == "2024-01-01"

        # Check doc2
        assert final_documents.iloc[1]["id"] == "doc2"
        assert final_documents.iloc[1]["text_unit_ids"] == ["unit3"]
        assert final_documents.iloc[1]["creation_date"] == "2024-01-02"

        # Verify all expected columns are present (including those that caused KeyError before)
        expected_columns = {
            "id", "human_readable_id", "title", "text", 
            "text_unit_ids", "creation_date", "metadata"
        }
        
        current_columns = set(final_documents.columns)
        missing_columns = expected_columns - current_columns
        assert not missing_columns, f"Missing columns in final documents: {missing_columns}"

    finally:
        # 6. Cleanup
        try:
            print(f"Cleaning up tables with prefix: {prefix}")
            client = storage._client
            instance = client.instance(spanner_config["instance_id"])
            database = instance.database(spanner_config["database_id"])
            # Drop the tables we created
            ddl = [
                f"DROP TABLE {prefix}documents",
                f"DROP TABLE {prefix}text_units"
            ]
            operation = database.update_ddl(ddl)
            operation.result(timeout=60)
            print("Cleanup complete.")
        except Exception as e:
            print(f"Warning: Cleanup failed for {prefix}: {e}")

@pytest.mark.asyncio
async def test_create_final_text_units_on_spanner(spanner_config):
    if not all(spanner_config.values()):
        pytest.skip("Spanner config not set")

    prefix = f"WFLTest_TU_{uuid4().hex[:8]}_"
    storage = SpannerPipelineStorage(**spanner_config, table_prefix=prefix)

    # Prepare initial data
    text_units_df = pd.DataFrame({
        "id": ["unit1", "unit2"],
        "text": ["chunk1", "chunk2"],
        "n_tokens": [10, 10],
        "document_ids": [["doc1"], ["doc1"]]
    })
    entities_df = pd.DataFrame({
        "id": ["ent1", "ent2"],
        "text_unit_ids": [["unit1"], ["unit1", "unit2"]]
    })
    relationships_df = pd.DataFrame({
        "id": ["rel1"],
        "text_unit_ids": [["unit1"]]
    })

    print(f"Creating initial tables with prefix: {prefix}")
    await storage.set_table("text_units", text_units_df)
    await storage.set_table("entities", entities_df)
    await storage.set_table("relationships", relationships_df)

    config = GraphRagConfig(
        models={
            defs.DEFAULT_CHAT_MODEL_ID: LanguageModelConfig(
                type=ModelType.Chat,
                model="mock-model",
                model_provider="openai",
                api_key="mock-api-key",
            ),
            defs.DEFAULT_EMBEDDING_MODEL_ID: LanguageModelConfig(
                type=ModelType.Embedding,
                model="mock-embedding-model",
                model_provider="openai",
                api_key="mock-api-key",
            ),
        }
    )
    stats = PipelineRunStats()
    context = PipelineRunContext(
        stats=stats,
        input_storage=MemoryPipelineStorage(),
        output_storage=storage,
        cache=InMemoryCache(),
        previous_storage=MemoryPipelineStorage(),
        callbacks=NoopWorkflowCallbacks(),
        state=PipelineState(),
    )

    try:
        print("Running create_final_text_units workflow...")
        await run_create_final_text_units(config, context)
        print("Workflow finished.")

        final_text_units = await storage.load_table("text_units")
        print("Loaded final text_units from Spanner.")
        
        assert len(final_text_units) == 2
        final_text_units = final_text_units.sort_values("id").reset_index(drop=True)

        # unit1 should have ent1, ent2, rel1
        unit1 = final_text_units.iloc[0]
        assert unit1["id"] == "unit1"
        # Spanner returns numpy arrays for ARRAY<STRING> sometimes? Or lists.
        # Let's convert to set for comparison.
        assert set(unit1["entity_ids"]) == {"ent1", "ent2"}
        assert set(unit1["relationship_ids"]) == {"rel1"}

        # unit2 should have ent2
        unit2 = final_text_units.iloc[1]
        assert unit2["id"] == "unit2"
        assert set(unit2["entity_ids"]) == {"ent2"}
        
        # Check relationship_ids for unit2. It might be None or empty list or NaN.
        rel_ids = unit2["relationship_ids"]
        # If it's None or NaN, it's acceptable as "no relationships"
        if rel_ids is None or (isinstance(rel_ids, float) and np.isnan(rel_ids)):
             pass
        else:
             # If it's a list, it should be empty
             assert len(rel_ids) == 0

    finally:
        # Cleanup
        try:
            print(f"Cleaning up tables with prefix: {prefix}")
            client = storage._client
            instance = client.instance(spanner_config["instance_id"])
            database = instance.database(spanner_config["database_id"])
            ddl = [
                f"DROP TABLE {prefix}text_units",
                f"DROP TABLE {prefix}entities",
                f"DROP TABLE {prefix}relationships"
            ]
            operation = database.update_ddl(ddl)
            operation.result(timeout=60)
            print("Cleanup complete.")
        except Exception:
            pass
