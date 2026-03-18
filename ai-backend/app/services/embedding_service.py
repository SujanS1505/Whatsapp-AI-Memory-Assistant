# FILE: ai-backend/app/services/embedding_service.py
from __future__ import annotations

from typing import List

from app.config import get_settings
from app.services.bedrock_client import invoke_model_async
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Generates vector embeddings using Amazon Titan Embed Text V2 via AWS Bedrock.
    Model: amazon.titan-embed-text-v2:0  (1024-dimension vectors)
    """

    def __init__(self):
        self.settings = get_settings()

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding vector for a single text string.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        body = {
            "inputText": text.strip(),
            "dimensions": 1024,
            "normalize": True,
        }
        try:
            logger.debug(f"Generating Bedrock embedding ({len(text)} chars)...")
            result = await invoke_model_async(self.settings.bedrock_embedding_model, body)
            embedding = result["embedding"]
            logger.debug(f"Embedding generated — {len(embedding)} dimensions.")
            return embedding
        except Exception as e:
            logger.error(f"Bedrock embedding failed: {e}")
            raise

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Bedrock Titan does not support batch in one call — runs sequentially.
        """
        cleaned = [t.strip() for t in texts if t and t.strip()]
        if not cleaned:
            return []

        results = []
        for text in cleaned:
            results.append(await self.embed_text(text))
        logger.info(f"Batch embedding complete — {len(results)} vectors.")
        return results
