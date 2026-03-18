# FILE: ai-backend/app/services/bedrock_client.py
from __future__ import annotations

import asyncio
import json
from functools import partial
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_bedrock_client = None


def get_bedrock_client():
    """Return a shared (lazy-initialized) boto3 bedrock-runtime client."""
    global _bedrock_client
    if _bedrock_client is None:
        settings = get_settings()
        _bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            config=Config(
                connect_timeout=10,
                read_timeout=60,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )
        logger.info(f"Bedrock client initialized in region {settings.aws_region}")
    return _bedrock_client


async def invoke_model_async(model_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Async wrapper around boto3's synchronous invoke_model.
    Runs in a thread executor so it doesn't block the event loop.
    """
    client = get_bedrock_client()
    payload = json.dumps(body)

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        partial(
            client.invoke_model,
            modelId=model_id,
            body=payload,
            contentType="application/json",
            accept="application/json",
        ),
    )
    return json.loads(response["body"].read())
