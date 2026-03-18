# FILE: ai-backend/app/database/message_repository.py

from datetime import datetime, timezone
import re
from typing import List, Optional
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import get_settings
from app.models.message_model import StoredMessage
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SEARCH_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "was", "were",
    "what", "when", "where", "who", "whom", "which", "how", "why", "can", "could", "would", "please",
    "tell", "about", "between", "with", "from", "that", "this", "have", "has", "had", "your", "their",
    "our", "my", "his", "her", "its",
}


class MessageRepository:
    def __init__(self, db: AsyncIOMotorDatabase):
        settings = get_settings()
        self.collection = db[settings.messages_collection]

    async def insert_message(self, message: StoredMessage) -> str:
        """
        Insert a message document into MongoDB.

        Returns:
            The string representation of the inserted document's ObjectId.
        """
        doc = {
            "group_id": message.group_id,
            "sender": message.sender,
            "message": message.message,
            "timestamp": message.timestamp,
            "embedding_id": message.embedding_id,
        }
        try:
            result = await self.collection.insert_one(doc)
            message_id = str(result.inserted_id)
            logger.info(f"Inserted message {message_id} from {message.sender} in group {message.group_id}")
            return message_id
        except Exception as e:
            logger.error(f"Failed to insert message: {e}")
            raise

    async def update_embedding_id(self, message_id: str, embedding_id: str) -> None:
        """Attach the ChromaDB vector ID to an existing message document."""
        try:
            await self.collection.update_one(
                {"_id": ObjectId(message_id)},
                {"$set": {"embedding_id": embedding_id}},
            )
            logger.debug(f"Updated embedding_id for message {message_id}")
        except Exception as e:
            logger.error(f"Failed to update embedding_id for {message_id}: {e}")
            raise

    async def get_messages_since(
        self, group_id: str, since: datetime
    ) -> List[StoredMessage]:
        """Retrieve all messages in a group since a given datetime."""
        try:
            cursor = self.collection.find(
                {
                    "group_id": group_id,
                    "timestamp": {"$gte": since},
                }
            ).sort("timestamp", 1)
            docs = await cursor.to_list(length=None)
            return [
                StoredMessage(
                    group_id=doc["group_id"],
                    sender=doc["sender"],
                    message=doc["message"],
                    timestamp=doc["timestamp"],
                    embedding_id=doc.get("embedding_id"),
                )
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"Failed to fetch messages since {since}: {e}")
            raise

    async def get_messages_by_ids(self, embedding_ids: List[str]) -> List[StoredMessage]:
        """Retrieve messages that match a list of embedding IDs."""
        try:
            cursor = self.collection.find(
                {"embedding_id": {"$in": embedding_ids}}
            ).sort("timestamp", 1)
            docs = await cursor.to_list(length=None)
            return [
                StoredMessage(
                    group_id=doc["group_id"],
                    sender=doc["sender"],
                    message=doc["message"],
                    timestamp=doc["timestamp"],
                    embedding_id=doc.get("embedding_id"),
                )
                for doc in docs
            ]
        except Exception as e:
            logger.error(f"Failed to fetch messages by embedding IDs: {e}")
            raise

    async def search_messages_by_text(
        self,
        group_id: str,
        text: str,
        limit: int = 50,
    ) -> List[StoredMessage]:
        """Keyword search fallback in MongoDB for a topic string."""
        raw = (text or "").strip()
        if not raw:
            return []

        tokens = [
            t for t in re.split(r"\W+", raw.lower())
            if len(t) >= 3 and t not in _SEARCH_STOPWORDS
        ]
        # Use OR-pattern over meaningful tokens for better recall than exact phrase matching.
        pattern = "|".join(re.escape(t) for t in tokens) if tokens else re.escape(raw)

        try:
            cursor = self.collection.find(
                {
                    "group_id": group_id,
                    "$or": [
                        {"message": {"$regex": pattern, "$options": "i"}},
                        {"sender": {"$regex": pattern, "$options": "i"}},
                    ],
                }
            ).sort("timestamp", -1).limit(limit)
            docs = await cursor.to_list(length=limit)
            messages = [
                StoredMessage(
                    group_id=doc["group_id"],
                    sender=doc["sender"],
                    message=doc["message"],
                    timestamp=doc["timestamp"],
                    embedding_id=doc.get("embedding_id"),
                )
                for doc in docs
            ]
            return messages[::-1]
        except Exception as e:
            logger.error(f"Failed keyword search for group {group_id}: {e}")
            raise

    async def ensure_indexes(self) -> None:
        """Create indexes for common query patterns."""
        await self.collection.create_index([("group_id", 1), ("timestamp", -1)])
        await self.collection.create_index([("embedding_id", 1)])
        logger.info("MongoDB indexes ensured on messages collection.")
