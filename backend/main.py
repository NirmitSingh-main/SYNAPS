"""
FastAPI application entry point for SYNAPS Signal Intelligence Backend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.signal import router as signal_router
from backend.api.analysis import router as analysis_router
from backend.api.report import router as report_router
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