"""
AI Settings API routes.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging

from ...database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai-settings", tags=["ai-settings"])


@router.get("/")
async def get_ai_settings(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Get user's AI settings.
    """
    headers = request.headers
    user_id = headers.get("user_id")

    if not user_id:
        raise HTTPException(status_code=401, detail="User authentication required")

    try:
        result = await db.execute(
            text("""
                SELECT ai_provider, ai_model, ai_api_key, ai_base_url, whisper_model, default_caption_lines
                FROM users
                WHERE id = :user_id
            """),
            {"user_id": user_id}
        )
        row = result.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="User not found")

        # Don't return API key in response for security
        return {
            "aiProvider": row.ai_provider or "local",
            "aiModel": row.ai_model or "llama3",
            "aiBaseUrl": row.ai_base_url or "http://host.docker.internal:11434/v1",
            "whisperModel": row.whisper_model or "medium",
            "defaultCaptionLines": getattr(row, 'default_caption_lines', 1),
            "hasApiKey": bool(row.ai_api_key)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving AI settings: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving AI settings: {str(e)}")


@router.patch("/")
async def update_ai_settings(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Update user's AI settings.
    """
    data = await request.json()
    headers = request.headers

    user_id = headers.get("user_id")

    if not user_id:
        raise HTTPException(status_code=401, detail="User authentication required")

    ai_provider = data.get("aiProvider")
    ai_model = data.get("aiModel")
    ai_api_key = data.get("aiApiKey")
    ai_base_url = data.get("aiBaseUrl")
    whisper_model = data.get("whisperModel")
    caption_lines = data.get("defaultCaptionLines")

    # Validate provider
    if ai_provider and ai_provider not in ["local", "openai", "google", "anthropic"]:
        raise HTTPException(status_code=400, detail="Invalid AI provider")

    # Validate whisper model
    if whisper_model and whisper_model not in ["tiny", "base", "small", "medium", "large"]:
        raise HTTPException(status_code=400, detail="Invalid Whisper model")

    # Validate caption lines
    if caption_lines is not None and caption_lines not in [1, 2, 3]:
        raise HTTPException(status_code=400, detail="Caption lines must be 1, 2, or 3")

    try:
        # Build update query
        update_fields = []
        params = {"user_id": user_id}

        if ai_provider is not None:
            update_fields.append("ai_provider = :ai_provider")
            params["ai_provider"] = ai_provider

        if ai_model is not None:
            update_fields.append("ai_model = :ai_model")
            params["ai_model"] = ai_model

        if ai_api_key is not None:
            update_fields.append("ai_api_key = :ai_api_key")
            params["ai_api_key"] = ai_api_key

        if ai_base_url is not None:
            update_fields.append("ai_base_url = :ai_base_url")
            params["ai_base_url"] = ai_base_url

        if whisper_model is not None:
            update_fields.append("whisper_model = :whisper_model")
            params["whisper_model"] = whisper_model

        if caption_lines is not None:
            update_fields.append("default_caption_lines = :default_caption_lines")
            params["default_caption_lines"] = caption_lines

        if update_fields:
            query = f"""
                UPDATE users
                SET {', '.join(update_fields)}
                WHERE id = :user_id
            """
            await db.execute(text(query), params)
            await db.commit()
            logger.info(f"Updated AI settings for user {user_id}")

        return {"message": "AI settings updated successfully"}

    except Exception as e:
        logger.error(f"Error updating AI settings: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating AI settings: {str(e)}")