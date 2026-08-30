"""
Pydantic Schemas for SYNAPS Signal Intelligence REST API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    file_path: Optional[str] = Field(None, description="Path to IQ or WAV file on server")
    sample_rate: Optional[float] = Field(None, description="Sampling rate in Hz (optional)")
    samples_per_symbol: int = Field(10, description="Nominal samples per symbol")


class ModulationPrediction(BaseModel):
    final_modulation: str
    decision_status: str
    confidence_pct: float
    detection_status: str
    probabilities: Dict[str, float]


class DspMetrics(BaseModel):
    snr_db: float
    carrier_frequency_offset_hz: float
    bandwidth_3db_hz: float
    symbol_rate: float
    peak_frequency_hz: float


class RecoveryPipelineSummary(BaseModel):
    synchronization_status: str
    recovered_symbols: int
    recovered_bits: int
    decoded_message: Optional[str] = None
    payload_entropy: float


class IntelligenceReportResponse(BaseModel):
    report_title: str
    signal_id: str
    source_file: Optional[str] = None
    format: Optional[str] = None
    sample_count: int
    sample_rate_hz: float
    modulation_decision: ModulationPrediction
    dsp_metrics: DspMetrics
    recovery_pipeline: RecoveryPipelineSummary
    emitter_fingerprint: Dict[str, Any]
    evidence_summary: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    service: str
    ai_available: bool
    version: str = "1.0.0"