# FILE: ai-backend/app/services/vector_service.py
from __future__ import annotations

import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_chroma_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        settings = get_settings()
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(f"ChromaDB client initialized at path: {settings.chroma_db_path}")
    return _chroma_client


class VectorService:
    """Stores and retrieves message embeddings using ChromaDB."""

    def __init__(self):
        self.settings = get_settings()
        client = get_chroma_client()
        self.collection = client.get_or_create_collection(
            name=self.settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug(f"Using ChromaDB collection: {self.settings.chroma_collection_name}")

    def store_vector(
        self,
        embedding: List[float],
        metadata: Dict[str, Any],
        embedding_id: str | None = None,
    ) -> str:
        """
        Persist an embedding vector alongside its metadata.

        Args:
            embedding: The vector to store.
            metadata: Arbitrary metadata (group_id, sender, timestamp, message snippet).
            embedding_id: Optional ID; generated automatically if not provided.

        Returns:
            The ID used to store the vector.
        """
        vec_id = embedding_id or str(uuid.uuid4())
        try:
            self.collection.add(
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[vec_id],
            )
            logger.debug(f"Stored vector {vec_id} in ChromaDB.")
            return vec_id
        except Exception as e:
            logger.error(f"Failed to store vector in ChromaDB: {e}")
            raise

    def similarity_search(
        self,
        query_embedding: List[float],
        group_id: str,
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform a similarity search restricted to a specific WhatsApp group.

        Args:
            query_embedding: Query vector.
            group_id: The WhatsApp group to scope the search.
            top_k: Number of top results to return. Defaults to config value.

        Returns:
            List of result dicts with keys: id, distance, metadata.
        """
        k = top_k or self.settings.top_k_results
        try:
            logger.debug(f"Similarity search in group {group_id}, top_k={k}")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where={"group_id": group_id},
                include=["metadatas", "distances"],
            )

            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            hits = []
            for vec_id, dist, meta in zip(ids, distances, metadatas):
                hits.append({"id": vec_id, "distance": dist, "metadata": meta})

            logger.info(f"Similarity search returned {len(hits)} results.")
            return hits
        except Exception as e:
            logger.error(f"ChromaDB similarity search failed: {e}")
            raise

    def delete_by_group(self, group_id: str) -> None:
        """Remove all vectors associated with a specific group."""
        try:
            self.collection.delete(where={"group_id": group_id})
            logger.info(f"Deleted all vectors for group {group_id}.")
        except Exception as e:
            logger.error(f"Failed to delete vectors for group {group_id}: {e}")
            raise
