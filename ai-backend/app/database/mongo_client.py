# FILE: ai-backend/app/database/mongo_client.py
from __future__ import annotations

from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_client: Optional[AsyncIOMotorClient] = None


async def connect_to_mongodb() -> None:
    """Establish connection to MongoDB."""
    global _client
    settings = get_settings()
    try:
        _client = AsyncIOMotorClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=5000,
        )
        # Ping to verify connectivity
        await _client.admin.command("ping")
        logger.info(f"Connected to MongoDB at {settings.mongodb_uri}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongodb_connection() -> None:
    """Close the MongoDB connection."""
    global _client
    if _client:
        _client.close()
        _client = None
        logger.info("MongoDB connection closed.")


def get_database() -> AsyncIOMotorDatabase:
    """Return the active database instance."""
    if _client is None:
        raise RuntimeError("MongoDB client is not initialized. Call connect_to_mongodb() first.")
    settings = get_settings()
    return _client[settings.mongodb_db_name]
