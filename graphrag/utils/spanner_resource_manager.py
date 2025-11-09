# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Manager for shared Spanner resources (Clients and Databases) to avoid orphaned threads."""

import logging
import threading
from typing import Any

from google.cloud import spanner

logger = logging.getLogger(__name__)


class SpannerResourceManager:
    """
    Singleton-like manager for Spanner resources.
    Ensures that only one spanner.Database object exists per logical database,
    sharing the underlying spanner.Client and Session Pool.
    """

    _clients: dict[str, spanner.Client] = {}
    _databases: dict[str, Any] = {} # Use Any to avoid importing Database class directly
    _client_ref_counts: dict[str, int] = {}
    _database_ref_counts: dict[str, int] = {}
    _lock = threading.Lock()

    @classmethod
    def get_database(
        cls,
        project_id: str,
        instance_id: str,
        database_id: str,
        credentials: Any = None,
    ) -> Any:
        """Get a shared Spanner Database object, creating it and its Client if necessary."""
        # Construct a unique key for the database
        db_key = f"projects/{project_id}/instances/{instance_id}/databases/{database_id}"
        client_key = project_id

        with cls._lock:
            if db_key not in cls._databases:
                logger.debug("Creating new shared Spanner Database for %s", db_key)
                
                # Get or create Client
                if client_key not in cls._clients:
                    logger.debug("Creating new shared Spanner Client for %s", client_key)
                    cls._clients[client_key] = spanner.Client(project=project_id, credentials=credentials)
                    cls._client_ref_counts[client_key] = 0
                
                cls._client_ref_counts[client_key] += 1
                client = cls._clients[client_key]
                
                # Create Database object
                instance = client.instance(instance_id)
                database = instance.database(database_id)
                cls._databases[db_key] = database
                cls._database_ref_counts[db_key] = 0

            cls._database_ref_counts[db_key] += 1
            logger.debug("Acquired shared Database for %s, ref_count=%d", db_key, cls._database_ref_counts[db_key])
            return cls._databases[db_key]

    @classmethod
    def release_database(cls, database: Any) -> None:
        """Release a Spanner Database, potentially closing its underlying Client."""
        db_key = database.name
        
        with cls._lock:
            if db_key not in cls._databases:
                logger.warning("Attempted to release unknown Database: %s", db_key)
                return

            cls._database_ref_counts[db_key] -= 1
            logger.debug("Released shared Database for %s, ref_count=%d", db_key, cls._database_ref_counts[db_key])

            if cls._database_ref_counts[db_key] <= 0:
                logger.debug("Removing shared Database %s from cache", db_key)

                # Attempt to explicitly shut down the multiplexed session maintenance thread.
                # Evidence from py-spy shows this thread ("maintenance-multiplexed-session-...")
                # can remain active and prevent clean process exit.
                try:
                    if hasattr(database, "_sessions_manager"):
                        mgr = database._sessions_manager
                        if hasattr(mgr, "_multiplexed_session_terminate_event"):
                            logger.debug(
                                "Setting termination event for multiplexed session maintenance thread on Database %s",
                                db_key,
                            )
                            mgr._multiplexed_session_terminate_event.set()
                except Exception as e:
                    logger.warning(
                        "Non-critical error while trying to terminate multiplexed session thread for %s: %s",
                        db_key,
                        e,
                    )

                del cls._databases[db_key]
                del cls._database_ref_counts[db_key]
                
                # Also release the associated Client
                # We need to parse project_id from db_key or store it. 
                # db_key format: projects/<project>/instances/<instance>/databases/<database>
                parts = db_key.split("/")
                if len(parts) >= 2 and parts[0] == "projects":
                    client_key = parts[1]
                    if client_key in cls._clients:
                        cls._client_ref_counts[client_key] -= 1
                        logger.debug("Decremented Client ref_count for %s to %d", client_key, cls._client_ref_counts[client_key])
                        
                        if cls._client_ref_counts[client_key] <= 0:
                            logger.debug("Closing shared Spanner Client for %s", client_key)
                            client_to_close = cls._clients.pop(client_key)
                            del cls._client_ref_counts[client_key]
                            client_to_close.close()
                else:
                    logger.warning("Could not parse project_id from database name: %s", db_key)