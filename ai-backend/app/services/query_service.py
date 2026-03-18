# FILE: ai-backend/app/services/query_service.py

import re
from datetime import timezone
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
_SEARCH_RE = re.compile(r"^search\s+(.+)$", re.IGNORECASE)
_SUMMARIZE_TOPIC_RE = re.compile(r"^summarize\s+(.+)$", re.IGNORECASE)

_QUERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are", "was", "were",
    "what", "when", "where", "who", "whom", "which", "how", "why", "can", "could", "would", "please",
    "tell", "about", "between", "with", "from", "that", "this", "have", "has", "had", "your", "their",
    "our", "my", "his", "her", "its",
}
_NOISE_PREFIXES = (
    "@assistant",
    "@chimera assistant",
    "*ai assistant:*",
    "sorry, i encountered an error while processing your query",
)


def _is_daily_summary_question(question: str) -> bool:
    """Detect daily-summary intent with tolerant phrasing."""
    q = question.strip().lower()
    if q in {"daily summary", "daily summry", "summarize today's discussion", "summarize todays discussion"}:
        return True
    if q.startswith("daily summ"):
        return True
    if q.startswith("today's summ") or q.startswith("todays summ"):
        return True
    return False


def _normalize_question(text: str) -> str:
    """Normalize common chat phrasing before command routing."""
    q = (text or "").strip()
    # Remove leading punctuation often present after bot mention, e.g. ", can you ..."
    q = re.sub(r"^[\s,.:;!\-]+", "", q)
    # Remove polite helper prefixes so intent routing works for natural language
    q = re.sub(r"^(can\s+you|could\s+you|would\s+you|please)\s+", "", q, flags=re.IGNORECASE)
    return q.strip()


def _extract_topic_from_summary_question(question: str) -> str | None:
    """Extract summary topic from natural language forms."""
    q = question.strip()

    # Prefer explicit topic phrasing first: "... on the topic <topic>"
    explicit = re.search(r"\bon\s+the\s+topic\s+(?:of\s+)?(.+)$", q, flags=re.IGNORECASE)
    if explicit:
        topic = explicit.group(1).strip(" .?!")
        if topic:
            return topic

    # Direct command style: "summarize <topic>"
    m = _SUMMARIZE_TOPIC_RE.match(q)
    if m:
        topic = m.group(1).strip(" .?!")
        return topic or None

    # Conversational style: "summarize ... about/on/regarding <topic>"
    patterns = [
        r"summarize(?:\s+the\s+conversation)?(?:\s+between\s+.+?)?\s+(?:about|on|regarding|for)\s+(.+)$",
        r"summarize.*?on\s+the\s+topic\s+(?:of\s+)?(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if not m:
            continue
        topic = m.group(1).strip(" .?!")
        topic = re.sub(r"^(the\s+topic\s+(?:of\s+)?)", "", topic, flags=re.IGNORECASE)
        if topic:
            return topic

    return None


def _format_context(messages: List[StoredMessage]) -> str:
    """Format retrieved messages as a numbered conversation context for the LLM."""
    lines = []
    for i, msg in enumerate(messages, start=1):
        ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
        lines.append(f"{i}. [{ts}] {msg.sender}: {msg.message}")
    return "\n".join(lines)


def _is_noise_message(text: str) -> bool:
    """Filter non-conversational lines that pollute retrieval quality."""
    lowered = (text or "").strip().lower()
    if not lowered:
        return True
    return any(lowered.startswith(prefix) for prefix in _NOISE_PREFIXES)


def _query_terms(question: str) -> List[str]:
    """Extract meaningful lexical terms for reranking context hits."""
    tokens = re.findall(r"[a-z0-9']+", (question or "").lower())
    return [t for t in tokens if len(t) >= 3 and t not in _QUERY_STOPWORDS]


def _lexical_score(msg: StoredMessage, terms: List[str]) -> int:
    """Simple token overlap score over sender + message text."""
    haystack = f"{msg.sender} {msg.message}".lower()
    return sum(1 for t in terms if t in haystack)


def _dedupe_messages(messages: List[StoredMessage]) -> List[StoredMessage]:
    """Remove duplicate messages while preserving order."""
    seen = set()
    unique: List[StoredMessage] = []
    for msg in messages:
        key = (msg.sender, msg.message, msg.timestamp.isoformat())
        if key in seen:
            continue
        seen.add(key)
        unique.append(msg)
    return unique


def _timestamp_for_sort(msg: StoredMessage) -> float:
    """Return a comparable epoch timestamp for both naive and aware datetimes."""
    ts = msg.timestamp
    if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts.timestamp()


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
        question = _normalize_question(request.question)

        # --- Route: daily summary ---
        if _is_daily_summary_question(question):
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
        topic = _extract_topic_from_summary_question(question)
        if topic:
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

        # Keep only semantically relevant matches. Lower cosine distance means higher similarity.
        relevant_hits = [
            hit
            for hit in hits
            if hit.get("distance") is not None and hit.get("distance") <= self.settings.retrieval_max_distance
        ]
        if not relevant_hits:
            return []

        # 3. Hydrate from metadata (avoids an extra MongoDB roundtrip for common queries)
        messages: List[StoredMessage] = []
        for hit in relevant_hits:
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

        # Exclude bot commands and assistant boilerplate from context.
        messages = [m for m in messages if not _is_noise_message(m.message)]

        # Hybrid rescue: Always merge lexical fallback from Mongo to ensure strong grounding
        # especially for recent facts or entity-heavy queries that vectors might miss.
        terms = _query_terms(question)
        if terms:
            fallback_limit = max(self.settings.top_k_results * 3, 30)
            fallback_messages = await self.repo.search_messages_by_text(
                group_id=group_id,
                text=question,
                limit=fallback_limit,
            )
            fallback_messages = [m for m in fallback_messages if not _is_noise_message(m.message)]
            if fallback_messages:
                logger.info(
                    "[QueryService] Retrieval used keyword fallback to improve grounding "
                    f"(fallback_count={len(fallback_messages)})"
                )
                messages = _dedupe_messages(messages + fallback_messages)

        if terms:
            # Sort by lexical match score first, then by recency.
            messages.sort(key=lambda m: (_lexical_score(m, terms), _timestamp_for_sort(m)), reverse=True)
            
        messages = messages[: self.settings.top_k_results]
        
        # Finally, sort chronologically so the LLM reads the context in natural chat order
        messages.sort(key=_timestamp_for_sort)

        return messages

    async def _rag_answer(self, group_id: str, question: str) -> QueryResponse:
        """Retrieve context and answer with RAG, with general-knowledge fallback."""
        context_messages = await self._retrieve_context(group_id, question)

        if not context_messages:
            logger.info("[QueryService] No relevant context found; using general-knowledge fallback.")
            answer_text = await self._general_answer(question)
            return QueryResponse(answer=answer_text, sources_count=0)

        context_text = _format_context(context_messages)

        system_prompt = (
            "You are an AI assistant that analyzes WhatsApp group discussions. "
            "Use the provided conversation context when it is relevant and sufficient. "
            "If the context is missing or insufficient to answer the user's question, "
            "fall back to your general knowledge and still provide a helpful answer. "
            "When falling back, briefly mention that the answer is not based on group history. "
            "Do not speculate when an explicit factual statement exists in context. "
            "For person-specific facts (for example job role), prefer direct statements from that person."
        )
        user_prompt = (
            f"Context — past group messages:\n\n{context_text}\n\n"
            f"User Question: {question}\n\n"
            "Generate a clear answer. Prefer context-grounded details where available."
        )
        prompt = _mistral_prompt(system_prompt, user_prompt)

        answer_text = await self._invoke_llm_text(prompt)
        logger.info(f"[QueryService] RAG answer generated, sources: {len(context_messages)}")
        return QueryResponse(answer=answer_text, sources_count=len(context_messages))

    async def _general_answer(self, question: str) -> str:
        """Answer directly with model knowledge when no conversation context is found."""
        system_prompt = (
            "You are a helpful AI assistant. "
            "Answer the user's question clearly and accurately using your general knowledge. "
            "Since no relevant WhatsApp history was found, mention that this answer is general "
            "knowledge and not derived from the group's conversation."
        )
        user_prompt = f"User Question: {question}"
        prompt = _mistral_prompt(system_prompt, user_prompt)
        return await self._invoke_llm_text(prompt)

    async def _invoke_llm_text(self, prompt: str) -> str:
        """Invoke Bedrock LLM and return text output."""

        try:
            body = {
                "prompt": prompt,
                "max_tokens": 1200,
                "temperature": 0.3,
            }
            result = await invoke_model_async(self.settings.bedrock_llm_model, body)
            return result["outputs"][0]["text"].strip()
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

        if not context_messages:
            fallback_limit = max(self.settings.top_k_results * 3, 30)
            context_messages = await self.repo.search_messages_by_text(
                group_id=group_id,
                text=topic,
                limit=fallback_limit,
            )
            if context_messages:
                logger.info(
                    f"[QueryService] Topic summarize used keyword fallback, found {len(context_messages)} messages"
                )

        summary = await self.summarizer.summarize_topic(group_id, topic, context_messages)
        return QueryResponse(answer=summary, sources_count=len(context_messages))
