"""
Signal management and inspection API endpoints.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile
from pathlib import Path
import shutil
import tempfile

from project_paths import DATA_ROOT, IQ_ROOT, WAV_ROOT, resolve_sample_paths
from signal_processing.input.loader import load_signal
from signal_processing.input.format_detection import detect_format

router = APIRouter(prefix="/signal", tags=["Signal"])


@router.post("/upload")
async def upload_signal(file: UploadFile = File(...)):
    """
    Upload an IQ or WAV signal file to the server for analysis.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".iq", ".wav"]:
        raise HTTPException(status_code=400, detail="Only .iq and .wav files are supported.")

    upload_dir = DATA_ROOT / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest_path = upload_dir / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "saved_path": str(dest_path),
        "format": suffix[1:].upper(),
        "size_bytes": dest_path.stat().st_size,
    }


@router.get("/inspect")
def inspect_signal(file_path: str):
    """
    Inspect basic physical parameters of a stored signal file.
    """
    path = Path(file_path)
    if not path.exists():
        # Try resolving via canonical resolver
        resolved = resolve_sample_paths(file_path)
        if resolved.get("iq_path") and resolved["iq_path"].exists():
            path = resolved["iq_path"]
        elif resolved.get("wav_path") and resolved["wav_path"].exists():
            path = resolved["wav_path"]
        else:
            raise HTTPException(status_code=404, detail=f"Signal file not found: {file_path}")

    try:
        samples, fs = load_signal(str(path))
        fmt = detect_format(str(path))
        return {
            "file_path": str(path),
            "filename": path.name,
            "format": fmt,
            "sample_count": len(samples),
            "sample_rate_hz": fs,
            "is_complex": True,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/samples")
def list_sample_signals():
    """
    List preloaded dataset sample signals for quick testing in the UI.
    """
    from project_paths import CLASS_NAMES, get_class_iq_dir, get_class_wav_dir
    samples_list = []

    for c in CLASS_NAMES:
        iq_dir = get_class_iq_dir(c)
        wav_dir = get_class_wav_dir(c)

        if iq_dir.exists():
            for f in sorted(list(iq_dir.glob("*.iq")))[:5]:
                samples_list.append({
                    "sample_id": f.stem,
                    "filename": f.name,
                    "modulation": c,
                    "format": "IQ",
                    "file_path": str(f),
                    "size_bytes": f.stat().st_size,
                })

        if wav_dir.exists():
            for f in sorted(list(wav_dir.glob("*.wav")))[:2]:
                samples_list.append({
                    "sample_id": f.stem,
                    "filename": f.name,
                    "modulation": c,
                    "format": "WAV",
                    "file_path": str(f),
                    "size_bytes": f.stat().st_size,
                })

    return {
        "count": len(samples_list),
        "samples": samples_list,
    }