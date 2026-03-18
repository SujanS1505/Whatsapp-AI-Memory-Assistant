# FILE: ai-backend/app/services/query_service.py

import re
from typing import List

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import get_settings
from app.database.message_repository import MessageRepository
from app.models.message_model import QueryRequest, QueryResponse, StoredMessage
from app.services.bedrock_client import invoke_model_async
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.summarization_service import SummarizationService, _mistral_prompt
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Regex patterns for special commands
# Matches "daily summary/summry/summ..." variations and common phrasings
_DAILY_SUMMARY_RE = re.compile(
    r"^(daily\s+summ\w*|summarize\s+today'?s?\s+discussion|today'?s?\s+summ\w*)$",
    re.IGNORECASE,
)
_SEARCH_RE = re.compile(r"^search\s+(.+)$", re.IGNORECASE)
_SUMMARIZE_TOPIC_RE = re.compile(r"^summarize\s+(.+)$", re.IGNORECASE)


def _format_context(messages: List[StoredMessage]) -> str:
    """Format retrieved messages as a numbered conversation context for the LLM."""
    lines = []
    for i, msg in enumerate(messages, start=1):
        ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
        lines.append(f"{i}. [{ts}] {msg.sender}: {msg.message}")
    return "\n".join(lines)


class QueryService:
    """
    Orchestrates query handling:
      - Routes special commands (daily summary, search, summarize topic)
      - Falls through to general RAG-based Q&A for all other queries
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.settings = get_settings()
        self.repo = MessageRepository(db)
        self.embedder = EmbeddingService()
        self.vector_store = VectorService()
        self.summarizer = SummarizationService(db)

    async def answer(self, request: QueryRequest) -> QueryResponse:
        """
        Dispatch the query to the appropriate handler.

        Supported patterns:
          - "daily summary" / "summarize today's discussion" → daily_summary
          - "search <topic>" → topic_search
          - "summarize <topic>" → topic_summarize
          - Anything else → general RAG Q&A

        Returns:
            QueryResponse with the AI-generated answer.
        """
        question = request.question.strip()

        # --- Route: daily summary ---
        if _DAILY_SUMMARY_RE.match(question):
            logger.info(f"[QueryService] Routing to daily summary for group {request.group_id}")
            answer = await self.summarizer.daily_summary(request.group_id)
            return QueryResponse(answer=answer, sources_count=0)

        # --- Route: topic search ---
        match = _SEARCH_RE.match(question)
        if match:
            topic = match.group(1).strip()
            logger.info(f"[QueryService] Routing to topic search: '{topic}'")
            return await self._topic_search(request.group_id, topic)

        # --- Route: summarize a topic ---
        match = _SUMMARIZE_TOPIC_RE.match(question)
        if match:
            topic = match.group(1).strip()
            logger.info(f"[QueryService] Routing to topic summarize: '{topic}'")
            return await self._topic_summarize(request.group_id, topic)

        # --- Default: general RAG Q&A ---
        logger.info(f"[QueryService] Routing to general RAG Q&A for: '{question}'")
        return await self._rag_answer(request.group_id, question)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _retrieve_context(self, group_id: str, question: str) -> List[StoredMessage]:
        """Embed the question, search vectors, and hydrate StoredMessage objects."""
        # 1. Generate embedding for the question
        query_embedding = await self.embedder.embed_text(question)

        # 2. Vector similarity search
        hits = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            group_id=group_id,
            top_k=self.settings.top_k_results,
        )

        if not hits:
            return []

        # 3. Hydrate from metadata (avoids an extra MongoDB roundtrip for common queries)
        messages: List[StoredMessage] = []
        for hit in hits:
            meta = hit.get("metadata", {})
            from datetime import datetime, timezone
            try:
                ts = datetime.fromisoformat(meta.get("timestamp", ""))
            except (ValueError, TypeError):
                ts = datetime.now(tz=timezone.utc)

            messages.append(
                StoredMessage(
                    group_id=meta.get("group_id", group_id),
                    sender=meta.get("sender", "Unknown"),
                    message=meta.get("message", ""),
                    timestamp=ts,
                    embedding_id=hit.get("id"),
                )
            )

        return messages

    async def _rag_answer(self, group_id: str, question: str) -> QueryResponse:
        """Retrieve context and ask the LLM for an answer."""
        context_messages = await self._retrieve_context(group_id, question)

        if not context_messages:
            return QueryResponse(
                answer="I couldn't find any relevant messages in this group's history to answer your question.",
                sources_count=0,
            )

        context_text = _format_context(context_messages)

        system_prompt = (
            "You are an AI assistant that analyzes WhatsApp group discussions. "
            "Answer the user's question clearly and concisely based strictly on the "
            "conversation context provided. If the context does not contain enough "
            "information, say so honestly. Do not invent facts."
        )
        user_prompt = (
            f"Context — past group messages:\n\n{context_text}\n\n"
            f"User Question: {question}\n\n"
            f"Generate a clear answer based on the discussion above."
        )
        prompt = _mistral_prompt(system_prompt, user_prompt)

        try:
            body = {
                "prompt": prompt,
                "max_tokens": 1200,
                "temperature": 0.3,
            }
            result = await invoke_model_async(self.settings.bedrock_llm_model, body)
            answer_text = result["outputs"][0]["text"].strip()
            logger.info(f"[QueryService] RAG answer generated, sources: {len(context_messages)}")
            return QueryResponse(answer=answer_text, sources_count=len(context_messages))
        except Exception as e:
            logger.error(f"[QueryService] Bedrock LLM call failed: {e}")
            raise

    async def _topic_search(self, group_id: str, topic: str) -> QueryResponse:
        """Return relevant discussion messages about a topic, formatted as a list."""
        context_messages = await self._retrieve_context(group_id, topic)

        if not context_messages:
            return QueryResponse(
                answer=f"No relevant messages found about '{topic}'.",
                sources_count=0,
            )

        lines = [f"*Messages related to '{topic}':*\n"]
        for msg in context_messages:
            ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(f"• [{ts}] *{msg.sender}*: {msg.message}")

        return QueryResponse(
            answer="\n".join(lines),
            sources_count=len(context_messages),
        )

    async def _topic_summarize(self, group_id: str, topic: str) -> QueryResponse:
        """Retrieve messages related to a topic and produce an LLM summary."""
        context_messages = await self._retrieve_context(group_id, topic)
        summary = await self.summarizer.summarize_topic(group_id, topic, context_messages)
        return QueryResponse(answer=summary, sources_count=len(context_messages))
