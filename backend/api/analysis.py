"""Signal analysis endpoints invoking the master pipeline service."""

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pathlib import Path
from typing import Optional
import shutil
import tempfile

from backend.schemas.response import AnalysisRequest
from backend.services.pipeline import analyze_signal
from project_paths import DATA_ROOT, resolve_sample_paths
from starlette.concurrency import run_in_threadpool

router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post("/run")
def run_analysis(request: AnalysisRequest):
    """
    Run complete end-to-end signal intelligence analysis on a target file path.
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
            raise HTTPException(
                status_code=404,
                detail=f"Signal file not found: {request.file_path}",
            )

    try:
        results = analyze_signal(
            file_path_or_samples=str(path),
            sample_rate=request.sample_rate,
            samples_per_symbol=request.samples_per_symbol,
        )

        return results.get("frontend_data", results.get("report"))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline failed: {e}",
        )


@router.post("/upload-and-analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    sample_rate: Optional[float] = Form(None),
    samples_per_symbol: int = Form(10),
):
    """
    Upload a signal file (.iq or .wav) and execute full analysis in a single step.
    Returns complete visualization and intelligence report structures for frontend.
    """
    import traceback
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse

    dest_path = None
    try:
        suffix = Path(file.filename).suffix.lower() if file.filename else ""

        if suffix not in [".iq", ".wav"]:
            raise HTTPException(
                status_code=400,
                detail="Only .iq and .wav files are supported.",
            )

        # Vercel's deployed filesystem is read-only.
        # /tmp is the writable temporary filesystem available to serverless functions.
        upload_dir = Path(tempfile.gettempdir()) / "synaps_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Keep only the filename component to prevent path traversal.
        safe_filename = Path(file.filename).name
        dest_path = upload_dir / safe_filename

        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = await run_in_threadpool(
            analyze_signal,
            file_path_or_samples=str(dest_path),
            sample_rate=sample_rate,
            samples_per_symbol=samples_per_symbol,
        )

        response_payload = results.get("frontend_data", results.get("report"))
        return JSONResponse(content=jsonable_encoder(response_payload))

    except HTTPException:
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[ERROR in upload_and_analyze]: {e}\n{error_trace}", flush=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "Signal analysis failed. Please verify the uploaded file and parameters.",
            },
        )
    finally:
        # Remove the temporary upload after analysis completes.
        if dest_path is not None:
            try:
                dest_path.unlink(missing_ok=True)
            except Exception:
                pass


@router.post("/sample")
def analyze_sample(
    sample_id: str = Form(...),
    sample_rate: Optional[float] = Form(None),
    samples_per_symbol: int = Form(10),
):
    """
    Analyze a preloaded dataset sample by its identifier or filename.
    """
    resolved = resolve_sample_paths(sample_id)
    target = resolved.get("iq_path") or resolved.get("wav_path")

    if not target or not target.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Sample '{sample_id}' not found in dataset.",
        )

    try:
        results = analyze_signal(
            file_path_or_samples=str(target),
            sample_rate=sample_rate,
            samples_per_symbol=samples_per_symbol,
        )

        return results.get("frontend_data", results.get("report"))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sample analysis failed: {e}",
        )