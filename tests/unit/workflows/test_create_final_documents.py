# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import unittest
import pandas as pd
import numpy as np
from graphrag.index.workflows.create_final_documents import create_final_documents

class TestCreateFinalDocuments(unittest.TestCase):
    def test_create_final_documents_basic(self):
        documents = pd.DataFrame({
            "id": ["doc1"],
            "text": ["some text"],
            "title": ["Title1"],
            "creation_date": ["2024-01-01"]
        })
        text_units = pd.DataFrame({
            "id": ["unit1", "unit2"],
            "document_ids": [["doc1"], ["doc1"]],
            "text": ["some", "text"],
            "n_tokens": [4, 4]
        })
        
        result = create_final_documents(documents, text_units)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["id"], "doc1")
        self.assertEqual(set(result.iloc[0]["text_unit_ids"]), {"unit1", "unit2"})

    def test_create_final_documents_empty_text_units(self):
        documents = pd.DataFrame({
            "id": ["doc1"],
            "text": ["some text"],
            "title": ["Title1"],
            "creation_date": ["2024-01-01"]
        })
        # Empty text_units but with correct columns
        text_units = pd.DataFrame(columns=["id", "document_ids", "text", "n_tokens"])
        
        # This might raise KeyError if not handled correctly
        result = create_final_documents(documents, text_units)
        self.assertIn("text_unit_ids", result.columns)
        self.assertEqual(len(result), 1)
        self.assertTrue(pd.isna(result.iloc[0]["text_unit_ids"]))

    def test_create_final_documents_no_matching_text_units(self):
        documents = pd.DataFrame({
            "id": ["doc1"],
            "text": ["some text"],
            "title": ["Title1"],
            "creation_date": ["2024-01-01"]
        })
        # text_units that don't match any document
        text_units = pd.DataFrame({
            "id": ["unit1"],
            "document_ids": [["doc_other"]],
            "text": ["chunk"],
            "n_tokens": [10]
        })
        
        result = create_final_documents(documents, text_units)
        self.assertIn("text_unit_ids", result.columns)
        self.assertEqual(len(result), 1)
        self.assertTrue(pd.isna(result.iloc[0]["text_unit_ids"]))

    def test_create_final_documents_empty_documents(self):
        documents = pd.DataFrame(columns=["id", "text", "title", "creation_date"])
        text_units = pd.DataFrame(columns=["id", "document_ids", "text", "n_tokens"])
        
        result = create_final_documents(documents, text_units)
        self.assertEqual(len(result), 0)
        # It should still have the columns
        self.assertIn("text_unit_ids", result.columns)