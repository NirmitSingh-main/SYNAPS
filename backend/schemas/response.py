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
    detected_components: Optional[List[str]] = None


class DspMetrics(BaseModel):
    snr_db: float
    carrier_frequency_offset_hz: float
    occupied_bandwidth_hz: float
    symbol_rate: float
    peak_frequency_hz: float


class RecoveryPipelineSummary(BaseModel):
    synchronization_status: str
    recovered_symbols: int
    recovered_bits: int
    decoded_message: Optional[str] = None
    decoding_status: Optional[str] = None
    fec_status: Optional[str] = "NOT_CONFIGURED"
    payload_entropy: float


class BitRecoverySummary(BaseModel):
    validation_status: str
    reference_bit_count: Optional[int] = None
    recovered_bit_count: Optional[int] = None
    matched_bit_count: Optional[int] = None
    bit_accuracy_pct: Optional[float] = None
    ber: Optional[float] = None
    component_recovery: Optional[List[Dict[str, Any]]] = None


class IntelligenceReportResponse(BaseModel):
    report_title: str
    signal_id: str
    source_file: Optional[str] = None
    format: Optional[str] = None
    sample_count: int
    sample_rate_hz: float
    signal_type: Optional[str] = "SINGLE"
    modulation_decision: ModulationPrediction
    dsp_metrics: DspMetrics
    recovery_pipeline: RecoveryPipelineSummary
    bit_recovery: Optional[BitRecoverySummary] = None
    emitter_fingerprint: Dict[str, Any]
    evidence_summary: Dict[str, Any]
    component_results: Optional[List[Dict[str, Any]]] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    ai_available: bool
    version: str = "1.0.0"