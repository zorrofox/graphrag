# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""All the steps to transform final entities."""

from uuid import uuid4

import numpy as np
import pandas as pd

from graphrag.data_model.schemas import COMMUNITY_REPORTS_FINAL_COLUMNS


def finalize_community_reports(
    reports: pd.DataFrame,
    communities: pd.DataFrame,
) -> pd.DataFrame:
    """All the steps to transform final community reports."""
    # Merge with communities to add shared fields
    community_reports = reports.merge(
        communities.loc[:, ["community", "parent", "children", "size", "period"]],
        on="community",
        how="left",
        copy=False,
    )

    community_reports["community"] = community_reports["community"].astype(int)
    community_reports["human_readable_id"] = community_reports["community"]
    community_reports["id"] = [uuid4().hex for _ in range(len(community_reports))]

    # Ensure all final columns exist
    for col in COMMUNITY_REPORTS_FINAL_COLUMNS:
        if col not in community_reports.columns:
            # Provide sensible defaults for missing columns
            if col in ["children", "entity_ids", "relationship_ids", "text_unit_ids"]:
                 community_reports[col] = np.empty((len(community_reports), 0)).tolist()
            elif col in ["size", "level", "rank"]:
                 community_reports[col] = 0
            else:
                 community_reports[col] = ""

    return community_reports.loc[
        :,
        COMMUNITY_REPORTS_FINAL_COLUMNS,
    ]
