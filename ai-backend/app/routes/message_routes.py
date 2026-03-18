# FILE: ai-backend/app/routes/message_routes.py

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database.mongo_client import get_database
from app.models.message_model import IncomingMessage, MessageResponse
from app.services.message_processor import MessageProcessor
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/messages", tags=["Messages"])


def get_message_processor(db: AsyncIOMotorDatabase = Depends(get_database)) -> MessageProcessor:
    return MessageProcessor(db)


@router.post(
    "",
    response_model=MessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a new group message",
)
async def ingest_message(
    payload: IncomingMessage,
    processor: MessageProcessor = Depends(get_message_processor),
) -> MessageResponse:
    """
    Receives a WhatsApp group message from the Node.js bot, stores it in
    MongoDB, generates an embedding, and persists the vector in ChromaDB.
    """
    logger.info(
        f"Received message from {payload.sender} in group {payload.group_id}: "
        f"{payload.message[:80]!r}"
    )
    try:
        message_id = await processor.process(payload)
        return MessageResponse(
            status="ok",
            message_id=message_id,
            detail="Message stored and embedded successfully.",
        )
    except Exception as e:
        logger.error(f"Message ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}",
        )
