# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for GCSStorage (v3 graphrag-storage package)."""

import re
import unittest
from datetime import datetime, timezone
from typing import cast
from unittest.mock import MagicMock, patch

from google.api_core.exceptions import NotFound, ServiceUnavailable, TooManyRequests
from graphrag_storage.gcs_storage import GCSStorage


class TestGCSStorage(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bucket_name = "test-bucket"
        self.base_dir = "test-dir"
        self.mock_client = MagicMock()
        self.mock_bucket = MagicMock()
        self.mock_client.bucket.return_value = self.mock_bucket

        with patch("google.cloud.storage.Client", return_value=self.mock_client):
            self.storage = GCSStorage(
                bucket_name=self.bucket_name, base_dir=self.base_dir
            )

    async def test_get_existing_key(self):
        key = "file.txt"
        content = b"hello world"
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.return_value = content
        self.mock_bucket.blob.return_value = mock_blob

        result = await self.storage.get(key, as_bytes=True)
        self.assertEqual(result, content)
        self.mock_bucket.blob.assert_called_with(f"{self.base_dir}/{key}")
        mock_blob.download_as_bytes.assert_called_once()

        result = await self.storage.get(key, as_bytes=False)
        self.assertEqual(result, content.decode("utf-8"))

    async def test_get_non_existent_key(self):
        key = "non_existent.txt"
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.side_effect = NotFound("not found")
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
        child_storage = cast("GCSStorage", self.storage.child(child_name))

        self.assertIsInstance(child_storage, GCSStorage)
        self.assertEqual(child_storage._base_dir, f"{self.base_dir}/{child_name}")
        self.assertEqual(child_storage._bucket_name, self.bucket_name)

    def test_find(self):
        """find() should yield relative blob names matching the pattern."""
        mock_blob1 = MagicMock()
        mock_blob1.name = f"{self.base_dir}/file1.txt"
        mock_blob2 = MagicMock()
        mock_blob2.name = f"{self.base_dir}/sub/file2.log"
        mock_blob3 = MagicMock()
        mock_blob3.name = f"{self.base_dir}/other.txt"

        self.mock_client.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]

        # v3: find() returns Iterator[str], not Iterator[tuple]
        pattern = re.compile(r".*\.txt$")
        results = list(self.storage.find(pattern))

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], "file1.txt")
        self.assertEqual(results[1], "other.txt")

    async def test_get_creation_date(self):
        key = "file.txt"
        mock_blob = MagicMock()
        dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_blob.time_created = dt
        self.mock_bucket.get_blob.return_value = mock_blob

        date_str = await self.storage.get_creation_date(key)
        self.assertIn("2023-01-01", date_str)

    async def test_delete_non_existent_key(self):
        mock_blob = MagicMock()
        mock_blob.delete.side_effect = NotFound("not found")
        self.mock_bucket.blob.return_value = mock_blob

        # Should not raise; NotFound is silently ignored for idempotent delete.
        await self.storage.delete("ghost.txt")
        mock_blob.delete.assert_called_once()

    def test_keys(self):
        mock_blob1 = MagicMock()
        mock_blob1.name = f"{self.base_dir}/file1.txt"
        mock_blob2 = MagicMock()
        mock_blob2.name = f"{self.base_dir}/subdir/file2.txt"
        self.mock_client.list_blobs.return_value = [mock_blob1, mock_blob2]

        keys = self.storage.keys()
        self.assertEqual(keys, ["file1.txt", "subdir/file2.txt"])

    def test_close(self):
        self.storage.close()
        self.mock_client.close.assert_called_once()

    def test_child_none_returns_self(self):
        result = self.storage.child(None)
        self.assertIs(result, self.storage)

    def test_child_shares_client(self):
        child = cast("GCSStorage", self.storage.child("sub-dir"))
        self.assertIs(child._client, self.storage._client)

    def test_find_empty_bucket(self):
        self.mock_client.list_blobs.return_value = []
        results = list(self.storage.find(re.compile(r".*\.txt$")))
        self.assertEqual(results, [])

    async def test_set_unsupported_type_raises(self):
        self.mock_bucket.blob.return_value = MagicMock()
        with self.assertRaises(TypeError):
            await self.storage.set("key.bin", 12345)

    async def test_get_exception_returns_none(self):
        mock_blob = MagicMock()
        mock_blob.download_as_bytes.side_effect = Exception("GCS error")
        self.mock_bucket.blob.return_value = mock_blob

        result = await self.storage.get("key.txt")
        self.assertIsNone(result)

    async def test_clear_empty_storage(self):
        self.mock_client.list_blobs.return_value = []
        await self.storage.clear()
        self.mock_client.list_blobs.assert_called_once()
        self.mock_bucket.delete_blobs.assert_not_called()

    async def test_get_creation_date_missing_blob(self):
        self.mock_bucket.get_blob.return_value = None
        result = await self.storage.get_creation_date("missing.txt")
        self.assertEqual(result, "")

    async def test_get_retries_on_transient_error(self):
        key = "file.txt"
        content = b"hello world"
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_blob.download_as_bytes.side_effect = [
            TooManyRequests("rate limited"),
            content,
        ]
        self.mock_bucket.blob.return_value = mock_blob

        result = await self.storage.get(key, as_bytes=True)
        self.assertEqual(result, content)
        self.assertEqual(mock_blob.download_as_bytes.call_count, 2)

    async def test_set_retries_on_service_unavailable(self):
        key = "file.txt"
        value = "hello world"
        mock_blob = MagicMock()
        mock_blob.upload_from_string.side_effect = [
            ServiceUnavailable("service down"),
            None,
        ]
        self.mock_bucket.blob.return_value = mock_blob

        await self.storage.set(key, value)
        self.assertEqual(mock_blob.upload_from_string.call_count, 2)

    def test_find_passes_page_size_to_list_blobs(self):
        self.mock_client.list_blobs.return_value = []
        list(self.storage.find(re.compile(r".*\.txt$")))
        call_kwargs = self.mock_client.list_blobs.call_args
        self.assertIsNone(call_kwargs.kwargs.get("max_results"))
        self.assertEqual(call_kwargs.kwargs.get("page_size"), 1000)

    def test_keys_passes_page_size_to_list_blobs(self):
        self.mock_client.list_blobs.return_value = []
        self.storage.keys()
        call_kwargs = self.mock_client.list_blobs.call_args
        self.assertIsNone(call_kwargs.kwargs.get("max_results"))
        self.assertEqual(call_kwargs.kwargs.get("page_size"), 1000)
