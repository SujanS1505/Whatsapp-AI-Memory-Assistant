# FILE: ai-backend/app/services/message_processor.py

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.message_model import IncomingMessage, StoredMessage
from app.database.message_repository import MessageRepository
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MessageProcessor:
    """
    Orchestrates the three-step ingestion pipeline:
      1. Store the message in MongoDB
      2. Generate an OpenAI embedding
      3. Persist the vector in ChromaDB and link it back to the MongoDB document
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.repo = MessageRepository(db)
        self.embedder = EmbeddingService()
        self.vector_store = VectorService()

    async def process(self, incoming: IncomingMessage) -> str:
        """
        Run the full ingestion pipeline for a single incoming message.

        Returns:
            The MongoDB ObjectId string of the stored message.
        """
        # Step 1 — Store in MongoDB (without embedding_id yet)
        stored = StoredMessage(
            group_id=incoming.group_id,
            sender=incoming.sender,
            message=incoming.message,
            timestamp=incoming.timestamp,
        )
        message_id = await self.repo.insert_message(stored)
        logger.info(f"[Pipeline] Step 1 complete — stored message {message_id}")

        # Step 2 — Generate embedding
        try:
            embedding = await self.embedder.embed_text(incoming.message)
            logger.info(f"[Pipeline] Step 2 complete — embedding generated for message {message_id}")
        except Exception as e:
            logger.error(f"[Pipeline] Embedding failed for message {message_id}: {e}")
            # Return the message_id anyway; vector will be missing but message is stored
            return message_id

        # Step 3 — Store in ChromaDB
        try:
            metadata = {
                "group_id": incoming.group_id,
                "sender": incoming.sender,
                "timestamp": incoming.timestamp.isoformat(),
                "message": incoming.message[:500],  # Store snippet in metadata for context
            }
            embedding_id = self.vector_store.store_vector(embedding, metadata)
            logger.info(f"[Pipeline] Step 3 complete — vector {embedding_id} stored in ChromaDB")
        except Exception as e:
            logger.error(f"[Pipeline] Vector storage failed for message {message_id}: {e}")
            return message_id

        # Link embedding_id back in Mongo
        try:
            await self.repo.update_embedding_id(message_id, embedding_id)
        except Exception as e:
            logger.warning(f"[Pipeline] Could not link embedding_id to MongoDB document: {e}")

        return message_id
