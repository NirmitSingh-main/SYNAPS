"""
Signal analysis endpoints invoking the master pipeline service.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path

from backend.schemas.response import AnalysisRequest, IntelligenceReportResponse
from backend.services.pipeline import analyze_signal
from project_paths import resolve_sample_paths

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("/run", response_model=IntelligenceReportResponse)
def run_analysis(request: AnalysisRequest):
    """
    Run complete end-to-end signal intelligence analysis on a target file.
    """
    if not request.file_path:
        raise HTTPException(status_code=400, detail="file_path must be specified.")

    path = Path(request.file_path)
    if not path.exists():
        resolved = resolve_sample_paths(request.file_path)
        if resolved.get("iq_path") and resolved["iq_path"].exists():
            path = resolved["iq_path"]
        elif resolved.get("wav_path") and resolved["wav_path"].exists():
            path = resolved["wav_path"]
        else:
            raise HTTPException(status_code=404, detail=f"Signal file not found: {request.file_path}")

    try:
        results = analyze_signal(
            file_path_or_samples=str(path),
            sample_rate=request.sample_rate,
            samples_per_symbol=request.samples_per_symbol,
        )
        return results["report"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline failed: {e}")