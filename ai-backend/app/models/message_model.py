# FILE: ai-backend/app/models/message_model.py

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class IncomingMessage(BaseModel):
    """Payload sent from the WhatsApp bot to the backend."""
    group_id: str = Field(..., description="WhatsApp group serialized ID")
    sender: str = Field(..., description="Display name or number of the sender")
    message: str = Field(..., description="Raw message text")
    timestamp: datetime = Field(..., description="ISO 8601 timestamp of when message was sent")


class StoredMessage(BaseModel):
    """Representation of a message as stored in MongoDB."""
    group_id: str
    sender: str
    message: str
    timestamp: datetime
    embedding_id: Optional[str] = None


class MessageResponse(BaseModel):
    """API response after storing a message."""
    status: str
    message_id: Optional[str] = None
    detail: Optional[str] = None


class QueryRequest(BaseModel):
    """Payload to query the AI assistant."""
    group_id: str = Field(..., description="WhatsApp group serialized ID")
    question: str = Field(..., description="User's natural language question")


class QueryResponse(BaseModel):
    """API response containing the AI-generated answer."""
    answer: str
    sources_count: int = 0
    detail: Optional[str] = None
