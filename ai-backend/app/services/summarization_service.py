# FILE: ai-backend/app/services/summarization_service.py

from datetime import datetime, timedelta, timezone
from typing import List

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import get_settings
from app.database.message_repository import MessageRepository
from app.models.message_model import StoredMessage
from app.services.bedrock_client import invoke_model_async
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _format_messages_for_prompt(messages: List[StoredMessage]) -> str:
    """Convert a list of StoredMessage objects into a readable conversation log."""
    lines = []
    for msg in messages:
        ts = msg.timestamp.strftime("%Y-%m-%d %H:%M")
        lines.append(f"[{ts}] {msg.sender}: {msg.message}")
    return "\n".join(lines)


def _mistral_prompt(system: str, user: str) -> str:
    """Format a prompt for Mistral Instruct chat format."""
    return f"<s>[INST] {system}\n\n{user} [/INST]"


async def _invoke_llm(prompt: str, max_tokens: int = 1500) -> str:
    """Call the Bedrock Mistral model and return the generated text."""
    settings = get_settings()
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }
    result = await invoke_model_async(settings.bedrock_llm_model, body)
    return result["outputs"][0]["text"].strip()


class SummarizationService:
    """Generates summaries of WhatsApp group discussions over a time window."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.settings = get_settings()
        self.repo = MessageRepository(db)

    async def daily_summary(self, group_id: str) -> str:
        """
        Summarize all messages from the last 24 hours in the given group.

        Returns:
            A natural-language summary string.
        """
        since = datetime.now(tz=timezone.utc) - timedelta(hours=24)
        messages = await self.repo.get_messages_since(group_id, since)

        if not messages:
            return "No messages found in the last 24 hours for this group."

        conversation = _format_messages_for_prompt(messages)
        logger.info(f"Summarizing {len(messages)} messages for group {group_id} (last 24h)")

        system_prompt = (
            "You are an AI assistant that summarizes WhatsApp group discussions. "
            "Provide a concise, structured daily summary that covers: "
            "key topics discussed, important decisions made, tasks assigned, "
            "and any notable mentions. Use bullet points where appropriate."
        )
        user_prompt = (
            f"Please provide a daily summary of the following WhatsApp group conversation:\n\n"
            f"{conversation}"
        )
        prompt = _mistral_prompt(system_prompt, user_prompt)

        try:
            summary = await _invoke_llm(prompt, max_tokens=1500)
            logger.info(f"Daily summary generated for group {group_id}.")
            return summary
        except Exception as e:
            logger.error(f"Bedrock LLM summarization failed: {e}")
            raise

    async def summarize_topic(self, group_id: str, topic: str, context_messages: List[StoredMessage]) -> str:
        """
        Summarize messages relevant to a specific topic.

        Args:
            group_id: WhatsApp group ID.
            topic: The topic to summarize.
            context_messages: Pre-retrieved relevant messages.

        Returns:
            A natural-language summary string.
        """
        if not context_messages:
            return f"No relevant messages found about '{topic}'."

        conversation = _format_messages_for_prompt(context_messages)
        logger.info(
            f"Summarizing {len(context_messages)} messages on topic '{topic}' for group {group_id}"
        )

        system_prompt = (
            "You are an AI assistant analyzing a WhatsApp group's past discussion. "
            "Focus on the specified topic and provide a clear, concise summary of "
            "what was discussed, who said what, and any conclusions or action items."
        )
        user_prompt = (
            f"Topic: {topic}\n\n"
            f"Relevant conversation excerpts:\n\n{conversation}\n\n"
            f"Summarize the discussion about '{topic}'."
        )
        prompt = _mistral_prompt(system_prompt, user_prompt)

        try:
            summary = await _invoke_llm(prompt, max_tokens=1000)
            logger.info(f"Topic summary generated for '{topic}' in group {group_id}.")
            return summary
        except Exception as e:
            logger.error(f"Bedrock LLM topic summarization failed: {e}")
            raise
