"""
SYNAPS Copilot — Supabase Auth & History Service
Verifies Supabase Bearer access tokens and retrieves authenticated user analysis history
using Supabase Auth and PostgREST with Row Level Security (RLS).
"""

from __future__ import annotations
import os
import logging
from typing import Any, Optional
from pathlib import Path
import httpx
from dotenv import load_dotenv

logger = logging.getLogger("synaps.copilot.auth")

# Ensure .env is loaded
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Also check frontend .env as fallback if needed
fe_env_path = Path(__file__).resolve().parent.parent.parent / "frontend" / "SNAPS-FE" / ".env"
if fe_env_path.exists():
    load_dotenv(dotenv_path=fe_env_path)


def get_supabase_config() -> tuple[str, str]:
    """Retrieve Supabase URL and Anon/Publishable Key from environment."""
    url = (
        os.getenv("SUPABASE_URL")
        or os.getenv("VITE_SUPABASE_URL")
        or "https://cxejugfvoywjqamchqne.supabase.co"
    ).rstrip("/")
    
    key = (
        os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY")
        or ""
    ).strip()
    
    return url, key


async def verify_supabase_token(access_token: str) -> Optional[dict[str, Any]]:
    """
    Verify a Supabase JWT access token by calling Supabase Auth endpoint:
    GET /auth/v1/user
    
    Returns the user dict if valid (containing 'id', 'email', etc.), or None if invalid/expired.
    """
    if not access_token or not access_token.strip():
        return None

    token = access_token.strip()
    supabase_url, supabase_key = get_supabase_config()

    if not supabase_url or not supabase_key:
        logger.warning("Supabase URL or key not configured in backend.")
        return None

    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {token}",
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(f"{supabase_url}/auth/v1/user", headers=headers)
            if resp.status_code == 200:
                user_data = resp.json()
                return user_data
            else:
                logger.warning(
                    f"Supabase auth verification failed ({resp.status_code}): {resp.text[:200]}"
                )
                return None
    except Exception as e:
        logger.error(f"Error communicating with Supabase Auth: {e}")
        return None


async def fetch_user_analyses(
    access_token: str,
    user_id: str,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """
    Fetch analysis_reports for the verified user using PostgREST.
    Passes the user's Bearer token so Supabase RLS is enforced at the DB level.
    Only queries lightweight metadata fields; skips large binary waveform arrays.
    """
    supabase_url, supabase_key = get_supabase_config()
    if not supabase_url or not supabase_key:
        return []

    fields = [
        "id",
        "user_id",
        "created_at",
        "filename",
        "format",
        "classification",
        "confidence",
        "sample_rate",
        "duration",
        "bandwidth",
        "snr",
        "peak_frequency",
        "num_samples",
        "prediction_breakdown",
        "features",
        "explanation",
    ]

    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {access_token.strip()}",
    }

    params = {
        "select": ",".join(fields),
        "user_id": f"eq.{user_id}",
        "order": "created_at.desc",
        "limit": str(limit),
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{supabase_url}/rest/v1/analysis_reports",
                headers=headers,
                params=params,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return data
            logger.warning(
                f"Supabase analysis_reports query returned {resp.status_code}: {resp.text[:200]}"
            )
            return []
    except Exception as e:
        logger.error(f"Error fetching user analysis reports from Supabase: {e}")
        return []
