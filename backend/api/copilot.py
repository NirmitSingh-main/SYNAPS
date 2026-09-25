"""
SYNAPS Copilot — FastAPI Router
POST /copilot/chat
"""

from __future__ import annotations
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field

from backend.copilot.service import get_copilot_service
from backend.copilot.auth import verify_supabase_token, fetch_user_analyses

router = APIRouter(prefix="/copilot", tags=["Copilot"])


# ── Request / Response models ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    page: Optional[str] = Field(None, description="Current page context (e.g. 'results', 'dashboard')")
    analysis_context: Optional[dict[str, Any]] = Field(None, description="Current analysis report data")
    history: Optional[list[ChatMessage]] = Field(None, description="Conversation history")
    # user_history is fetched server-side when needed — never accepted from client
    # to prevent injection of other users' data


class ChatResponse(BaseModel):
    response: str
    intent: list[str]
    model: str
    page: str
    success: bool


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def copilot_chat(
    request: ChatRequest,
    authorization: Optional[str] = Header(None),
) -> ChatResponse:
    """
    SYNAPS Copilot chat endpoint.
    Accepts user message with optional page/analysis context and conversation history.
    Authenticates Supabase Bearer token server-side to safely fetch user analysis history.
    Returns the Groq-powered assistant response.
    """
    service = get_copilot_service()

    # Authenticate and fetch user history server-side if token provided
    user_history: Optional[list[dict[str, Any]]] = None
    if authorization and authorization.strip():
        token = authorization
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if token:
            user_data = await verify_supabase_token(token)
            if user_data and "id" in user_data:
                user_id = user_data["id"]
                user_history = await fetch_user_analyses(token, user_id=user_id, limit=30)

    # Convert history messages to plain dicts for the service
    history_dicts: list[dict[str, str]] | None = None
    if request.history:
        history_dicts = [
            {"role": m.role, "content": m.content}
            for m in request.history
            if m.role in ("user", "assistant")
        ]

    result = service.chat(
        message=request.message,
        page=request.page,
        analysis_context=request.analysis_context,
        history=history_dicts,
        user_history=user_history,
    )

    return ChatResponse(**result)


@router.get("/status")
async def copilot_status() -> dict[str, Any]:
    """Check if Copilot (Groq) is configured and available."""
    service = get_copilot_service()
    return {
        "available": service.is_available(),
        "service": "SYNAPS Copilot",
    }
