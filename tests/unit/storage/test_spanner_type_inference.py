# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

# I will implement the inference logic in the storage class later, 
# but for now I'll define it here to TDD it, then move it.
def infer_spanner_type(series: pd.Series) -> str:
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
    if isinstance(first_val, bool): # Sometimes bools are in object columns
        return "BOOL"
    if isinstance(first_val, (int, np.integer)):
        return "INT64"
    if isinstance(first_val, (float, np.floating)):
        return "FLOAT64"

    if isinstance(first_val, (list, tuple, np.ndarray)):
        # Inspect first non-empty list if possible
        for val in non_null:
            if len(val) > 0:
                first_elem = val[0]
                if isinstance(first_elem, str):
                    return "ARRAY<STRING(MAX)>"
                if isinstance(first_elem, (int, np.integer)):
                    return "ARRAY<INT64>"
                if isinstance(first_elem, (float, np.floating)):
                    return "ARRAY<FLOAT64>"
                # If list contains complex objects, use JSON
                if isinstance(first_elem, (dict, list, tuple)):
                    return "JSON"
                break
        # If all lists are empty, default to JSON or ARRAY<STRING(MAX)>?
        # JSON is safer as it can hold empty lists.
        return "JSON"

    if isinstance(first_val, dict):
        return "JSON"
        
    return "STRING(MAX)"

class TestSpannerTypeInference(unittest.TestCase):
    def test_integers(self):
        s = pd.Series([1, 2, 3])
        self.assertEqual(infer_spanner_type(s), "INT64")
        
        s_nullable = pd.Series([1, None, 3], dtype="Int64")
        self.assertEqual(infer_spanner_type(s_nullable), "INT64")

    def test_floats(self):
        s = pd.Series([1.1, 2.2, None])
        self.assertEqual(infer_spanner_type(s), "FLOAT64")

    def test_bools(self):
        s = pd.Series([True, False])
        self.assertEqual(infer_spanner_type(s), "BOOL")

    def test_strings(self):
        s = pd.Series(["a", "b", None])
        self.assertEqual(infer_spanner_type(s), "STRING(MAX)")

    def test_timestamps(self):
        s = pd.Series([datetime.now(), None])
        self.assertEqual(infer_spanner_type(s), "TIMESTAMP")
        
        s_pd = pd.to_datetime(pd.Series(["2024-01-01", None]))
        self.assertEqual(infer_spanner_type(s_pd), "TIMESTAMP")

    def test_arrays(self):
        s_str_array = pd.Series([["a", "b"], ["c"], None])
        self.assertEqual(infer_spanner_type(s_str_array), "ARRAY<STRING(MAX)>")

        s_int_array = pd.Series([[1, 2], [3], []])
        self.assertEqual(infer_spanner_type(s_int_array), "ARRAY<INT64>")

        s_empty_arrays = pd.Series([[], [], None])
        # Should fallback to JSON if we can't determine inner type
        self.assertEqual(infer_spanner_type(s_empty_arrays), "JSON")

    def test_json(self):
        s_dict = pd.Series([{"a": 1}, {"b": 2}, None])
        self.assertEqual(infer_spanner_type(s_dict), "JSON")

        s_complex_list = pd.Series([[{"a": 1}], [{"b": 2}]])
        self.assertEqual(infer_spanner_type(s_complex_list), "JSON")

    def test_all_null(self):
        s = pd.Series([None, None])
        self.assertEqual(infer_spanner_type(s), "STRING(MAX)")
