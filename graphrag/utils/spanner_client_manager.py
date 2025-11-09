# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Manager for shared Spanner clients to avoid orphaned threads."""

import logging
import threading
from typing import Any

from google.cloud import spanner

logger = logging.getLogger(__name__)


class SpannerClientManager:
    """
    Singleton-like manager for Spanner clients.
    Ensures that only one spanner.Client exists per project_id (or globally if project_id is same),
    and it is only closed when all users of it have released it.
    """

    _clients: dict[str, spanner.Client] = {}
    _ref_counts: dict[str, int] = {}
    _lock = threading.Lock()

    @classmethod
    def get_client(cls, project_id: str | None = None, credentials: Any = None) -> spanner.Client:
        """Get a shared Spanner client, incrementing its reference count."""
        key = project_id or "default"
        with cls._lock:
            if key not in cls._clients:
                logger.debug("Creating new shared Spanner client for project_id=%s", key)
                cls._clients[key] = spanner.Client(project=project_id, credentials=credentials)
                cls._ref_counts[key] = 0
            
            cls._ref_counts[key] += 1
            logger.debug("Acquired shared Spanner client for project_id=%s, ref_count=%d", key, cls._ref_counts[key])
            return cls._clients[key]

    @classmethod
    def release_client(cls, client: spanner.Client) -> None:
        """Release a Spanner client, decrementing its reference count and closing it if 0."""
        with cls._lock:
            # Find the key for this client
            key_to_remove = None
            for key, managed_client in cls._clients.items():
                if managed_client is client:
                    cls._ref_counts[key] -= 1
                    logger.debug("Released shared Spanner client for project_id=%s, ref_count=%d", key, cls._ref_counts[key])
                    if cls._ref_counts[key] <= 0:
                        key_to_remove = key
                    break
            
            if key_to_remove:
                logger.debug("Closing shared Spanner client for project_id=%s as ref_count is 0", key_to_remove)
                client_to_close = cls._clients.pop(key_to_remove)
                del cls._ref_counts[key_to_remove]
                # Actually close the client.
                client_to_close.close()
