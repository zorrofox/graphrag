# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

import re
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from graphrag.storage.gcs_pipeline_storage import GCSPipelineStorage


class TestGCSPipelineStorage(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bucket_name = "test-bucket"
        self.base_dir = "test-dir"
        self.mock_client = MagicMock()
        self.mock_bucket = MagicMock()
        self.mock_client.bucket.return_value = self.mock_bucket

        with patch("google.cloud.storage.Client", return_value=self.mock_client):
            self.storage = GCSPipelineStorage(
                bucket_name=self.bucket_name, base_dir=self.base_dir
            )

    async def test_get_existing_key(self):
        key = "file.txt"
        content = b"hello world"
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = content
        self.mock_bucket.blob.return_value = mock_blob

        # Test getting as bytes
        result = await self.storage.get(key, as_bytes=True)
        self.assertEqual(result, content)
        self.mock_bucket.blob.assert_called_with(f"{self.base_dir}/{key}")
        mock_blob.download_as_bytes.assert_called_once()

        # Test getting as string
        result = await self.storage.get(key, as_bytes=False)
        self.assertEqual(result, content.decode("utf-8"))

    async def test_get_non_existent_key(self):
        key = "non_existent.txt"
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        self.mock_bucket.blob.return_value = mock_blob

        result = await self.storage.get(key)
        self.assertIsNone(result)

    async def test_set_string(self):
        key = "file.txt"
        value = "hello world"
        mock_blob = MagicMock()
        self.mock_bucket.blob.return_value = mock_blob

        await self.storage.set(key, value)
        self.mock_bucket.blob.assert_called_with(f"{self.base_dir}/{key}")
        mock_blob.upload_from_string.assert_called_with(
            value, content_type="text/plain; charset=utf-8"
        )

    async def test_set_bytes(self):
        key = "file.bin"
        value = b"\x00\x01\x02"
        mock_blob = MagicMock()
        self.mock_bucket.blob.return_value = mock_blob

        await self.storage.set(key, value)
        mock_blob.upload_from_string.assert_called_with(
            value, content_type="application/octet-stream"
        )

    async def test_has(self):
        key = "file.txt"
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        self.mock_bucket.blob.return_value = mock_blob

        self.assertTrue(await self.storage.has(key))

        mock_blob.exists.return_value = False
        self.assertFalse(await self.storage.has(key))

    async def test_delete(self):
        key = "file.txt"
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        self.mock_bucket.blob.return_value = mock_blob

        await self.storage.delete(key)
        mock_blob.delete.assert_called_once()

    async def test_clear(self):
        mock_blob1 = MagicMock()
        mock_blob2 = MagicMock()
        self.mock_client.list_blobs.return_value = [mock_blob1, mock_blob2]

        await self.storage.clear()
        
        self.mock_client.list_blobs.assert_called_with(
            self.bucket_name, prefix=f"{self.base_dir}/"
        )
        self.mock_bucket.delete_blobs.assert_called_with([mock_blob1, mock_blob2])

    def test_child(self):
        child_name = "sub-dir"
        with patch("google.cloud.storage.Client", return_value=self.mock_client):
            child_storage = self.storage.child(child_name)
        
        self.assertIsInstance(child_storage, GCSPipelineStorage)
        self.assertEqual(child_storage._base_dir, f"{self.base_dir}/{child_name}")
        self.assertEqual(child_storage._bucket_name, self.bucket_name)

    def test_find(self):
        mock_blob1 = MagicMock()
        mock_blob1.name = f"{self.base_dir}/file1.txt"
        mock_blob2 = MagicMock()
        mock_blob2.name = f"{self.base_dir}/sub/file2.log"
        mock_blob3 = MagicMock()
        mock_blob3.name = f"{self.base_dir}/other.txt"

        self.mock_client.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]

        # Find all .txt files
        pattern = re.compile(r".*\.txt$")
        results = list(self.storage.find(pattern))
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "file1.txt")
        self.assertEqual(results[1][0], "other.txt")

    async def test_get_creation_date(self):
        key = "file.txt"
        mock_blob = MagicMock()
        # Use a fixed datetime for testing
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_blob.time_created = dt
        self.mock_bucket.get_blob.return_value = mock_blob

        date_str = await self.storage.get_creation_date(key)
        # The format depends on get_timestamp_formatted_with_local_tz implementation
        # Assuming it returns something like "%Y-%m-%d %H:%M:%S %z"
        self.assertIn("2023-01-01", date_str)
