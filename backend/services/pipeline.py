"""
Master Signal Intelligence Analysis Pipeline Service.

Integrates:
  Input Format Detection -> Signal Loading -> Preprocessing ->
  DSP Spectral/Temporal/Statistical Analysis -> Synchronization ->
  AI Neural Network Classification -> Multi-modal Feature Fusion ->
  Modulation Decision -> Demodulation & Symbol Slicing ->
  Decoding & Payload Extraction -> RF Fingerprinting -> Intelligence Report.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

from project_paths import (
    CLASS_NAMES,
    normalize_modulation_name,
    resolve_sample_paths,
)
from signal_processing.input.format_detection import detect_format
from signal_processing.input.loader import load_signal
from signal_processing.input.validation import validate_samples
from signal_processing.detection.signal_detector import detect_signal
from signal_processing.preprocessing.dc_removal import remove_dc
from signal_processing.preprocessing.normalization import normalize_signal

# DSP modules
from dsp.spectral.fft import compute_fft
from dsp.spectral.psd import estimate_psd
from dsp.spectral.bandwidth import estimate_bandwidth
from dsp.signal_quality.snr import estimate_snr
from dsp.frequency.cfo import estimate_cfo
from dsp.frequency.frequency_estimation import estimate_frequency
from dsp.phase.phase_estimation import estimate_phase
from dsp.timing.symbol_rate import estimate_symbol_rate
from dsp.constellation.constellation import analyze_constellation
from dsp.statistical.hoc import calculate_hoc

# Synchronization
from signal_processing.synchronization.frequency_sync import correct_frequency_offset
from signal_processing.synchronization.phase_sync import correct_phase_offset
from signal_processing.synchronization.timing_sync import (
    estimate_timing_offset,
    sample_symbols,
)

# Demodulation & Decoding
from signal_processing.demodulation.bpsk import demodulate_bpsk
from signal_processing.demodulation.qpsk import demodulate_qpsk
from signal_processing.demodulation.fsk import demodulate_fsk
from signal_processing.demodulation.qam import demodulate_16qam
from signal_processing.decoding.bit_to_data import bits_to_data

# AI classification
from ai.inference.predict import (
    load_model,
    prepare_features,
    CLASS_NAMES as AI_CLASSES,
)
from ai.classification.confidence import calculate_confidence, confidence_percent
from ai.classification.unknown_detection import get_detection_status
from ai.classification.modulation import classify_modulation

# Intelligence & Fusion
from fusion.feature_fusion import fuse_dsp_and_ai_features
from fusion.evidence import aggregate_evidence
from intelligence.decision.hypothesis import generate_hypotheses
from intelligence.decision.decision import make_modulation_decision
from intelligence.confidence.confidence import compute_composite_confidence
from intelligence.detection.anomaly import detect_anomalies
from intelligence.fingerprint.fingerprint import extract_signal_fingerprint
from intelligence.validation.decoded_data import validate_decoded_payload
from intelligence.validation.recovery_validation import validate_recovery_pipeline
from intelligence.report.report import generate_intelligence_report


class AnalysisPipeline:
    """
    End-to-end signal intelligence analysis engine.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        try:
            self.model, self.device = load_model(model_path)
            self.ai_available = True
        except Exception as e:
            print(f"[WARN] AI Model could not be loaded: {e}. Running DSP-only mode.")
            self.model = None
            self.device = torch.device("cpu")
            self.ai_available = False

    def run(
        self,
        file_path_or_samples: Union[str, Path, np.ndarray],
        sample_rate: Optional[float] = None,
        samples_per_symbol: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end analysis on an IQ/WAV file or raw NumPy signal.
        """
        # 1. INPUT HANDLING
        if isinstance(file_path_or_samples, (str, Path)):
            input_path = Path(file_path_or_samples)
            fmt = detect_format(str(input_path))
            raw_samples, fs = load_signal(str(input_path), iq_sample_rate=sample_rate)
            sample_id = input_path.stem
            file_str = str(input_path)
        else:
            raw_samples = np.asarray(file_path_or_samples, dtype=np.complex64)
            fs = float(sample_rate) if sample_rate else 1_000_000.0
            fmt = "IQ"
            sample_id = "in_memory_signal"
            file_str = "in_memory"

        # 2. VALIDATION
        validate_samples(raw_samples, fs)

        # 3. SIGNAL DETECTION / SEGMENTATION
        try:
            detected_samples, region = detect_signal(raw_samples)
        except Exception:
            detected_samples = raw_samples
            region = (0, len(raw_samples))

        # 4. PREPROCESSING
        dc_clean = remove_dc(detected_samples)
        preprocessed = normalize_signal(dc_clean)

        # 5. DSP ANALYSIS
        fft_res = compute_fft(preprocessed, sampling_rate=fs)
        psd_res = estimate_psd(preprocessed, sampling_rate=fs)
        bw_res = estimate_bandwidth(preprocessed, sampling_rate=fs)
        snr_res = estimate_snr(preprocessed)
        freq_res = estimate_frequency(preprocessed, sampling_rate=fs)
        est_freq = float(freq_res.get("estimated_frequency_hz", 0.0))
        cfo_res = estimate_cfo(preprocessed, sampling_rate=fs, reference_frequency_hz=0.0)
        phase_res = estimate_phase(preprocessed, sampling_rate=fs, frequency_hz=est_freq)
        timing_res = estimate_symbol_rate(preprocessed, sampling_rate=fs)
        constellation_res = analyze_constellation(preprocessed)
        hoc_res = calculate_hoc(preprocessed)

        dsp_summary = {
            "snr_db": float(snr_res.get("snr_db", 0.0)),
            "cfo_hz": float(cfo_res.get("cfo_hz", 0.0)),
            "bandwidth_hz": float(bw_res.get("bandwidth_3db_hz", bw_res.get("bandwidth_hz", 0.0))),
            "symbol_rate": float(timing_res.get("symbol_rate", 0.0)),
            "peak_frequency_hz": float(fft_res.get("peak_frequency_hz", 0.0)),
            "phase_offset_rad": float(phase_res.get("phase_offset_radians", 0.0)),
            "hoc": hoc_res,
            "constellation": constellation_res,
        }

        # 6. SYNCHRONIZATION
        est_cfo = dsp_summary["cfo_hz"]
        est_phase_deg = float(phase_res.get("phase_offset_degrees", 0.0))
        freq_synced = correct_frequency_offset(preprocessed, est_cfo, fs)
        phase_synced = correct_phase_offset(freq_synced, est_phase_deg)

        try:
            timing_offset = estimate_timing_offset(phase_synced, samples_per_symbol)
            symbols = sample_symbols(phase_synced, samples_per_symbol, timing_offset)
            sync_status = "SUCCESS"
        except Exception:
            symbols = phase_synced
            timing_offset = 0
            sync_status = "DEGRADED"

        # 7. AI CLASSIFICATION
        if self.ai_available and self.model is not None:
            features = prepare_features(preprocessed)
            x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x_tensor)
            probs, pred_idx, conf = calculate_confidence(logits)
            pred_class = classify_modulation(pred_idx)
            conf_pct = confidence_percent(conf)
            det_status = get_detection_status(conf)

            prob_dict = {
                name: float(probs[i].item() * 100.0) for i, name in enumerate(AI_CLASSES)
            }
        else:
            pred_class = "UNKNOWN"
            conf_pct = 50.0
            det_status = "UNKNOWN"
            prob_dict = {name: 25.0 for name in CLASS_NAMES}

        ai_summary = {
            "predicted_class": pred_class,
            "confidence": conf_pct,
            "status": det_status,
            "probabilities": prob_dict,
        }

        # 8. FUSION & EVIDENCE
        fused = fuse_dsp_and_ai_features(dsp_summary, ai_summary)
        evidence = aggregate_evidence(fused)

        # 9. DECISION & COMPOSITE CONFIDENCE
        decision_res = make_modulation_decision(pred_class, conf_pct, evidence)
        composite_conf = compute_composite_confidence(conf_pct, dsp_summary["snr_db"])
        anomalies = detect_anomalies(dsp_summary, conf_pct)
        fingerprint = extract_signal_fingerprint(
            preprocessed, dsp_summary, decision_res["final_modulation"]
        )

        # 10. MODULATION-SPECIFIC DEMODULATION
        final_mod = decision_res["final_modulation"]
        recovered_bits = np.array([], dtype=np.uint8)

        try:
            if final_mod == "BPSK":
                recovered_bits = demodulate_bpsk(symbols)
            elif final_mod == "QPSK":
                recovered_bits = demodulate_qpsk(symbols)
            elif final_mod == "FSK":
                recovered_bits = demodulate_fsk(phase_synced, samples_per_symbol)
            elif final_mod == "QAM16":
                recovered_bits = demodulate_16qam(symbols)
            else:
                recovered_bits = demodulate_bpsk(symbols)
        except Exception as e:
            print(f"[WARN] Demodulation error for {final_mod}: {e}")
            recovered_bits = np.array([], dtype=np.uint8)

        # 11. DECODING & PAYLOAD VALIDATION
        decoded_text = None
        if len(recovered_bits) >= 8:
            try:
                decoded_text = bits_to_data(recovered_bits, encoding="ASCII")
            except Exception:
                decoded_text = None

        payload_val = validate_decoded_payload(decoded_text or "", recovered_bits)
        recovery_val = validate_recovery_pipeline(
            {"status": sync_status},
            {"recovered_bits": recovered_bits},
            payload_val,
        )

        # 12. COMPILE REPORT
        raw_analysis = {
            "input_info": {
                "sample_id": sample_id,
                "file_path": file_str,
                "format": fmt,
                "sample_count": len(raw_samples),
                "sample_rate_hz": fs,
                "active_region": region,
            },
            "dsp_analysis": dsp_summary,
            "synchronization": {
                "status": sync_status,
                "estimated_cfo_hz": est_cfo,
                "estimated_phase_deg": est_phase_deg,
                "timing_offset": timing_offset,
            },
            "ai_classification": ai_summary,
            "fused_features": fused,
            "evidence": evidence,
            "decision": decision_res,
            "confidence_assessment": composite_conf,
            "anomalies": anomalies,
            "fingerprint": fingerprint,
            "demodulation": {
                "modulation": final_mod,
                "symbol_count": len(symbols),
                "bit_count": len(recovered_bits),
                "recovered_bits": recovered_bits,
            },
            "decoding": {
                "decoded_message": decoded_text,
                "entropy": payload_val.get("entropy", 0.0),
                "printable_ratio": payload_val.get("printable_ratio", 0.0),
            },
            "recovery_validation": recovery_val,
        }

        report = generate_intelligence_report(raw_analysis)
        raw_analysis["report"] = report
        return raw_analysis


# Global default service instance
default_pipeline = AnalysisPipeline()


def analyze_signal(file_path_or_samples, sample_rate=None, samples_per_symbol=10, **kwargs):
    """
    Convenience function to analyze any signal via the default pipeline.
    """
    return default_pipeline.run(
        file_path_or_samples,
        sample_rate=sample_rate,
        samples_per_symbol=samples_per_symbol,
        **kwargs,
    )