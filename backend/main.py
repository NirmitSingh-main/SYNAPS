"""
FastAPI application entry point for SYNAPS Signal Intelligence Backend.
"""

import sys
import types
from pathlib import Path

# Register backend and synaps_core in sys.path for self-contained execution
_BACKEND_DIR = Path(__file__).resolve().parent
_SYNAPS_CORE_DIR = _BACKEND_DIR / "synaps_core"
_PROJECT_ROOT = _BACKEND_DIR.parent

for _p in (_PROJECT_ROOT, _BACKEND_DIR, _SYNAPS_CORE_DIR):
    if _p.exists():
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

# Read allowed origins from environment; default to production frontend + local dev.
_raw_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "https://synaps-black.vercel.app,http://localhost:5173",
)
_ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

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

# CORS middleware for frontend access.
# NOTE: allow_credentials=True requires explicit origins (not "*").
# Set FRONTEND_ORIGINS env var (comma-separated) to add more origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://synaps-black.vercel.app",
        "https://synapsintelligence.netlify.app",
    ],
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