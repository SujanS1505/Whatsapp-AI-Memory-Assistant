# FILE: ai-backend/app/routes/query_routes.py

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.database.mongo_client import get_database
from app.models.message_model import QueryRequest, QueryResponse
from app.services.query_service import QueryService
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


def get_query_service(db: AsyncIOMotorDatabase = Depends(get_database)) -> QueryService:
    return QueryService(db)


@router.post(
    "",
    response_model=QueryResponse,
    summary="Query the AI assistant about past group discussions",
)
async def query_assistant(
    payload: QueryRequest,
    service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    """
    Accepts a natural language question about a specific WhatsApp group's
    history, retrieves relevant context via vector similarity search, and
    returns an LLM-generated answer.
    """
    logger.info(f"Query for group {payload.group_id}: {payload.question!r}")
    try:
        result = await service.answer(payload)
        return result
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}",
        )
