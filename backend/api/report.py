"""
Intelligence report retrieval and export endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pathlib import Path

from backend.services.pipeline import analyze_signal
from intelligence.report.report import format_text_report
from project_paths import resolve_sample_paths

router = APIRouter(prefix="/report", tags=["Report"])


@router.get("/text", response_class=PlainTextResponse)
def get_text_report(file_path: str = Query(..., description="Target signal path")):
    """
    Generate and return a human-readable text intelligence report.
    """
    path = Path(file_path)
    if not path.exists():
        resolved = resolve_sample_paths(file_path)
        if resolved.get("iq_path") and resolved["iq_path"].exists():
            path = resolved["iq_path"]
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    try:
        res = analyze_signal(str(path))
        text = format_text_report(res["report"])
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))