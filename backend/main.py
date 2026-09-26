"""
FastAPI application entry point for SYNAPS Signal Intelligence Backend.
"""

import sys
import types
from pathlib import Path

# Robustly locate repository root containing project_paths.py across Vercel serverless and local environments
def _resolve_project_root() -> Path:
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path("/vercel/path0"),
    ]
    for cand in candidates:
        curr = cand.resolve()
        for _ in range(6):
            if (curr / "project_paths.py").exists():
                return curr
            if curr.parent == curr:
                break
            curr = curr.parent
    return Path(__file__).resolve().parent.parent

_PROJECT_ROOT = _resolve_project_root()
_BACKEND_DIR = _PROJECT_ROOT / "backend" if (_PROJECT_ROOT / "backend").exists() else Path(__file__).resolve().parent

for _p in (_PROJECT_ROOT, _BACKEND_DIR):
    _p_str = str(_p)
    if _p_str in sys.path:
        sys.path.remove(_p_str)
    sys.path.insert(0, _p_str)

# Ensure 'backend' package is resolvable in sys.modules
if "backend" not in sys.modules:
    try:
        import importlib
        importlib.import_module("backend")
    except ModuleNotFoundError:
        _backend_pkg = types.ModuleType("backend")
        _backend_pkg.__path__ = [str(_BACKEND_DIR)]
        _backend_pkg.__file__ = str(_BACKEND_DIR / "__init__.py")
        sys.modules["backend"] = _backend_pkg

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.signal import router as signal_router
from backend.api.analysis import router as analysis_router
from backend.api.report import router as report_router
from backend.api.copilot import router as copilot_router
from backend.schemas.response import HealthResponse
from backend.services.pipeline import default_pipeline

app = FastAPI(
    title="SYNAPS Signal Intelligence API",
    description="Backend API for automatic modulation recognition, DSP analysis, signal recovery, and RF fingerprinting.",
    version="1.0.0",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(signal_router)
app.include_router(analysis_router)
app.include_router(report_router)
app.include_router(copilot_router)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Service health and status endpoint.
    """
    return HealthResponse(
        status="ONLINE",
        service="SYNAPS Signal Intelligence Platform",
        ai_available=default_pipeline.ai_available,
        version="1.0.0",
    )


@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Welcome to SYNAPS Signal Intelligence API",
        "documentation": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)