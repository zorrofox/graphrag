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

    async def test_delete_non_existent_key(self):
        """Deleting a non-existent key should not call blob.delete()."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        self.mock_bucket.blob.return_value = mock_blob

        await self.storage.delete("ghost.txt")

        mock_blob.delete.assert_not_called()

    def test_keys(self):
        """keys() should return relative paths stripped of base_dir."""
        mock_blob1 = MagicMock()
        mock_blob1.name = f"{self.base_dir}/file1.txt"
        mock_blob2 = MagicMock()
        mock_blob2.name = f"{self.base_dir}/subdir/file2.txt"
        self.mock_client.list_blobs.return_value = [mock_blob1, mock_blob2]

        keys = self.storage.keys()

        self.assertEqual(keys, ["file1.txt", "subdir/file2.txt"])

    def test_close(self):
        """close() should invoke client.close()."""
        self.storage.close()
        self.mock_client.close.assert_called_once()

    def test_child_none_returns_self(self):
        """child(None) should return the same instance."""
        result = self.storage.child(None)
        self.assertIs(result, self.storage)

    def test_child_shares_client(self):
        """child() should reuse the parent's GCS client, not create a new one."""
        child = self.storage.child("sub-dir")

        self.assertIs(child._client, self.storage._client)

    def test_find_with_max_count(self):
        """find() should stop yielding after max_count matches."""
        blobs = []
        for i in range(5):
            b = MagicMock()
            b.name = f"{self.base_dir}/file{i}.txt"
            blobs.append(b)
        self.mock_client.list_blobs.return_value = blobs

        results = list(self.storage.find(re.compile(r".*\.txt$"), max_count=3))

        self.assertEqual(len(results), 3)

    def test_find_empty_bucket(self):
        """find() on an empty bucket yields nothing."""
        self.mock_client.list_blobs.return_value = []

        results = list(self.storage.find(re.compile(r".*\.txt$")))

        self.assertEqual(results, [])

    async def test_set_unsupported_type_raises(self):
        """set() with an unsupported value type should raise TypeError."""
        self.mock_bucket.blob.return_value = MagicMock()

        with self.assertRaises(TypeError):
            await self.storage.set("key.bin", 12345)

    async def test_get_exception_returns_none(self):
        """get() should return None when the GCS client raises."""
        mock_blob = MagicMock()
        mock_blob.exists.side_effect = Exception("GCS error")
        self.mock_bucket.blob.return_value = mock_blob

        result = await self.storage.get("key.txt")

        self.assertIsNone(result)

    async def test_clear_empty_storage(self):
        """clear() with no blobs should list blobs but skip delete_blobs."""
        self.mock_client.list_blobs.return_value = []

        await self.storage.clear()

        self.mock_client.list_blobs.assert_called_once()
        self.mock_bucket.delete_blobs.assert_not_called()

    async def test_get_creation_date_missing_blob(self):
        """get_creation_date() returns empty string when blob doesn't exist."""
        self.mock_bucket.get_blob.return_value = None

        result = await self.storage.get_creation_date("missing.txt")

        self.assertEqual(result, "")
