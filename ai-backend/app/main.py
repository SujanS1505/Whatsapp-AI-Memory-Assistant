# FILE: ai-backend/app/main.py

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database.mongo_client import connect_to_mongodb, close_mongodb_connection
from app.routes import message_routes, query_routes
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    # Startup
    logger.info("Starting WhatsApp AI Memory Assistant backend...")
    await connect_to_mongodb()
    logger.info("All services initialized. Ready to accept requests.")
    yield
    # Shutdown
    logger.info("Shutting down backend...")
    await close_mongodb_connection()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="WhatsApp Group AI Memory Assistant",
    description=(
        "An AI-powered backend that stores WhatsApp group messages, "
        "generates embeddings, and answers natural language queries about "
        "past discussions using retrieval-augmented generation (RAG)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — restrict in production to the actual bot host
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(message_routes.router)
app.include_router(query_routes.router)


@app.get("/", tags=["Health"])
async def root():
    return {"status": "ok", "service": "WhatsApp AI Memory Assistant"}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred."},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
