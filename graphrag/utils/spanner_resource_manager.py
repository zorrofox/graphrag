# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Manager for shared Spanner resources (Clients and Databases) to avoid orphaned threads."""

import logging
import os
import threading
from typing import Any

from google.auth.credentials import AnonymousCredentials
from google.cloud import spanner

logger = logging.getLogger(__name__)


def _build_spanner_client(project_id: str, credentials: Any) -> spanner.Client:
    """Create a Spanner client, using the local emulator if SPANNER_EMULATOR_HOST is set."""
    emulator_host = os.environ.get("SPANNER_EMULATOR_HOST")
    if emulator_host:
        logger.info(
            "Connecting to Spanner emulator at %s (project=%s)", emulator_host, project_id
        )
        return spanner.Client(project=project_id, credentials=AnonymousCredentials())
    return spanner.Client(project=project_id, credentials=credentials)


class SpannerResourceManager:
    """Singleton-like manager for Spanner resources.

    Ensures that only one ``spanner.Database`` object exists per logical
    database, sharing the underlying ``spanner.Client`` and session pool.

    Client keying includes the identity of the *credentials* object so that
    two callers using different service accounts for the same project each get
    their own client (previously, only the project-id was used as the key,
    which caused the wrong credentials to be reused).
    """

    _clients: dict[str, spanner.Client] = {}
    _databases: dict[str, Any] = {}
    _client_ref_counts: dict[str, int] = {}
    _database_ref_counts: dict[str, int] = {}
    # Maps db_key → client_key so release_database() does not need to parse
    # db_key to reconstruct the client key.
    _db_to_client_key: dict[str, str] = {}
    _lock = threading.Lock()

    @staticmethod
    def _make_client_key(project_id: str, credentials: Any) -> str:
        """Return a cache key that is unique per (project, credentials) pair.

        Uses the service-account email when available, otherwise falls back to
        the object's ``id()`` (stable for the lifetime of the credentials
        object within a single process).
        """
        if credentials is None:
            creds_id = "adc"  # application-default credentials
        elif hasattr(credentials, "service_account_email"):
            creds_id = credentials.service_account_email
        else:
            creds_id = str(id(credentials))
        return f"{project_id}/{creds_id}"

    @classmethod
    def get_database(
        cls,
        project_id: str,
        instance_id: str,
        database_id: str,
        credentials: Any = None,
    ) -> Any:
        """Get a shared Spanner Database object, creating it and its Client if necessary."""
        db_key = f"projects/{project_id}/instances/{instance_id}/databases/{database_id}"
        client_key = cls._make_client_key(project_id, credentials)

        with cls._lock:
            if db_key not in cls._databases:
                logger.debug("Creating new shared Spanner Database for %s", db_key)

                if client_key not in cls._clients:
                    logger.debug("Creating new shared Spanner Client for %s", client_key)
                    cls._clients[client_key] = _build_spanner_client(project_id, credentials)
                    cls._client_ref_counts[client_key] = 0

                cls._client_ref_counts[client_key] += 1
                client = cls._clients[client_key]

                instance = client.instance(instance_id)
                database = instance.database(database_id)
                cls._databases[db_key] = database
                cls._database_ref_counts[db_key] = 0
                cls._db_to_client_key[db_key] = client_key  # store mapping

            cls._database_ref_counts[db_key] += 1
            logger.debug(
                "Acquired shared Database for %s, ref_count=%d",
                db_key,
                cls._database_ref_counts[db_key],
            )
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
            logger.debug(
                "Released shared Database for %s, ref_count=%d",
                db_key,
                cls._database_ref_counts[db_key],
            )

            if cls._database_ref_counts[db_key] <= 0:
                logger.debug("Removing shared Database %s from cache", db_key)

                # Attempt to shut down the multiplexed-session maintenance thread
                # so the process can exit cleanly (evidence from py-spy).
                try:
                    if hasattr(database, "_sessions_manager"):
                        mgr = database._sessions_manager
                        if hasattr(mgr, "_multiplexed_session_terminate_event"):
                            logger.debug(
                                "Setting termination event for multiplexed session "
                                "maintenance thread on Database %s",
                                db_key,
                            )
                            mgr._multiplexed_session_terminate_event.set()
                except Exception as e:
                    logger.warning(
                        "Non-critical error while terminating multiplexed session "
                        "thread for %s: %s",
                        db_key,
                        e,
                    )

                del cls._databases[db_key]
                del cls._database_ref_counts[db_key]

                # Look up the client key from the stored mapping instead of
                # parsing it from db_key (which would not work if the key
                # format ever changes, or if it includes credentials info).
                client_key = cls._db_to_client_key.pop(db_key, None)
                if client_key and client_key in cls._clients:
                    cls._client_ref_counts[client_key] -= 1
                    logger.debug(
                        "Decremented Client ref_count for %s to %d",
                        client_key,
                        cls._client_ref_counts[client_key],
                    )
                    if cls._client_ref_counts[client_key] <= 0:
                        logger.debug(
                            "Closing shared Spanner Client for %s", client_key
                        )
                        client_to_close = cls._clients.pop(client_key)
                        del cls._client_ref_counts[client_key]
                        client_to_close.close()
                elif not client_key:
                    logger.warning(
                        "No client_key mapping found for database %s", db_key
                    )