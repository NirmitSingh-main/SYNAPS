"""
SYNAPS Copilot — LLM Service
Handles Groq SDK initialization, message formatting, API error handling,
and chat completion requests.
"""

from __future__ import annotations
import os
import logging
from typing import Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from backend.copilot.intents import classify_intent
from backend.copilot.prompts import build_copilot_messages

logger = logging.getLogger("synaps.copilot")

# Ensure .env is loaded from backend directory
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Candidate models in order of priority
MODELS_PRIORITY = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
]

DEFAULT_TIMEOUT = 25.0


class CopilotService:
    def __init__(self):
        self._api_key = os.getenv("GROQ_API_KEY", "").strip()
        self._client = None
        self._init_client()

    def _init_client(self):
        if not self._api_key:
            # Refresh from environment in case it was loaded later
            self._api_key = os.getenv("GROQ_API_KEY", "").strip()

        if self._api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self._api_key, timeout=DEFAULT_TIMEOUT)
                logger.info("SYNAPS Copilot: Groq client initialized successfully.")
            except Exception as e:
                logger.error(f"SYNAPS Copilot: Failed to initialize Groq client: {e}")
                self._client = None
        else:
            logger.warning("SYNAPS Copilot: GROQ_API_KEY is not set.")
            self._client = None

    def is_available(self) -> bool:
        """Check if Groq client is configured and ready."""
        if not self._client and os.getenv("GROQ_API_KEY"):
            self._init_client()
        return self._client is not None

    def chat(
        self,
        message: str,
        page: Optional[str] = None,
        analysis_context: Optional[dict[str, Any]] = None,
        history: Optional[list[dict[str, str]]] = None,
        user_history: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """
        Process a user message with full context, intent classification, and LLM response.
        """
        message = (message or "").strip()
        if not message:
            return {
                "response": "Please enter a question or message.",
                "intent": "EMPTY",
                "model": "none",
                "page": page or "general",
                "success": False,
            }

        # Classify intent for RAG context targeting
        intent = classify_intent(message, page)

        if not self.is_available():
            logger.error("SYNAPS Copilot: Groq client unavailable or missing API key.")
            return {
                "response": "Copilot is temporarily unavailable. Please verify the backend configuration.",
                "intent": intent,
                "model": "none",
                "page": page or "general",
                "success": False,
            }

        # Build prompt messages
        try:
            messages = build_copilot_messages(
                user_message=message,
                page=page,
                analysis=analysis_context,
                user_history=user_history,
                conversation_history=history,
            )
        except Exception as e:
            logger.error(f"SYNAPS Copilot: Error assembling prompt context: {e}")
            messages = [
                {"role": "system", "content": "You are SYNAPS Copilot, an RF Signal Intelligence assistant."},
                {"role": "user", "content": message},
            ]

        # Call Groq API with fallback models
        last_error = None
        for model in MODELS_PRIORITY:
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=600,
                    temperature=0.2,  # Low temperature for technical precision & consistency
                )
                text = response.choices[0].message.content or ""
                text = text.strip()
                if not text:
                    continue

                return {
                    "response": text,
                    "intent": intent,
                    "model": model,
                    "page": page or "general",
                    "success": True,
                }
            except Exception as e:
                logger.warning(f"SYNAPS Copilot: Call failed for model {model}: {e}")
                last_error = e

        logger.error(f"SYNAPS Copilot: All Groq models failed. Last error: {last_error}")
        return {
            "response": "Copilot is temporarily unavailable. Please try again in a moment.",
            "intent": intent,
            "model": "failed",
            "page": page or "general",
            "success": False,
        }


# Singleton service instance
_copilot_service: Optional[CopilotService] = None


def get_copilot_service() -> CopilotService:
    global _copilot_service
    if _copilot_service is None:
        _copilot_service = CopilotService()
    return _copilot_service
