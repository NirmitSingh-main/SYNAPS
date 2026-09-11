"""
Master Signal Intelligence Analysis Pipeline Service.

Integrates:
  Input Detection -> Loading -> Preprocessing ->
  AI Classification -> DSP Analysis -> Synchronization ->
  Feature Fusion -> Decision -> Demodulation ->
  Decoding -> RF Fingerprinting -> Intelligence Report.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import datetime
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

# DSP
from dsp.spectral.fft import compute_fft
from dsp.spectral.psd import estimate_psd
from dsp.spectral.bandwidth import estimate_bandwidth
from dsp.signal_quality.snr import estimate_snr
from dsp.frequency.cfo import (
    estimate_cfo,
    estimate_fsk_carrier_center,
)
from dsp.frequency.frequency_estimation import estimate_frequency
from dsp.phase.phase_estimation import estimate_phase
from dsp.timing.symbol_rate import estimate_symbol_rate
from dsp.constellation.constellation import analyze_constellation
from dsp.statistical.hoc import calculate_hoc

# Synchronization
from signal_processing.synchronization.frequency_sync import (
    correct_frequency_offset,
)
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

# AI
from ai.inference.predict import (
    load_model,
    prepare_features,
    CLASS_NAMES as AI_CLASSES,
)
from ai.classification.confidence import (
    calculate_confidence,
    confidence_percent,
)
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
from intelligence.validation.recovery_validation import (
    validate_recovery_pipeline,
)
from intelligence.report.report import generate_intelligence_report


class AnalysisPipeline:
    """End-to-end signal intelligence analysis engine."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
    ):
        try:
            self.model, self.device = load_model(model_path)
            self.ai_available = True
        except Exception as e:
            print(
                f"[WARN] AI Model could not be loaded: "
                f"{e}. Running DSP-only mode."
            )
            self.model = None
            self.device = torch.device("cpu")
            self.ai_available = False

    def run(
        self,
        file_path_or_samples: Union[str, Path, np.ndarray],
        sample_rate: Optional[float] = None,
        samples_per_symbol: int = 10,
    ) -> Dict[str, Any]:

        # =================================================================
        # 1. INPUT HANDLING
        # =================================================================

        if isinstance(file_path_or_samples, (str, Path)):
            input_path = Path(file_path_or_samples)
            fmt = detect_format(str(input_path))

            raw_samples, fs = load_signal(
                str(input_path),
                iq_sample_rate=sample_rate,
            )

            sample_id = input_path.stem
            file_str = str(input_path)

        else:
            raw_samples = np.asarray(
                file_path_or_samples,
                dtype=np.complex64,
            )
            fs = float(sample_rate) if sample_rate else 1_000_000.0
            fmt = "IQ"
            sample_id = "in_memory_signal"
            file_str = "in_memory"

        # =================================================================
        # 2. VALIDATION
        # =================================================================

        validate_samples(raw_samples, fs)

        # =================================================================
        # 3. SIGNAL DETECTION
        # =================================================================

        try:
            detected_samples, region = detect_signal(raw_samples)
        except Exception:
            detected_samples = raw_samples
            region = (0, len(raw_samples))

        # =================================================================
        # 4. PREPROCESSING
        # =================================================================

        dc_clean = remove_dc(detected_samples)
        preprocessed = normalize_signal(dc_clean)

        # =================================================================
        # 5. AI CLASSIFICATION
        # =================================================================

        if self.ai_available and self.model is not None:
            features = prepare_features(preprocessed)

            x_tensor = torch.tensor(
                features,
                dtype=torch.float32,
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(x_tensor)

            probs, pred_idx, conf = calculate_confidence(logits)
            pred_class = classify_modulation(pred_idx)

            conf_pct = confidence_percent(conf)
            det_status = get_detection_status(conf)

            prob_dict = {
                name: float(probs[i].item() * 100.0)
                for i, name in enumerate(AI_CLASSES)
            }

        else:
            pred_class = "UNKNOWN"
            conf_pct = 50.0
            det_status = "UNKNOWN"
            prob_dict = {
                name: 25.0
                for name in CLASS_NAMES
            }

        ai_summary = {
            "predicted_class": pred_class,
            "confidence": conf_pct,
            "status": det_status,
            "probabilities": prob_dict,
        }

        # =================================================================
        # 6. DSP ANALYSIS
        # =================================================================

        fft_res = compute_fft(
            preprocessed,
            sampling_rate=fs,
        )

        psd_res = estimate_psd(
            preprocessed,
            sampling_rate=fs,
        )

        bw_res = estimate_bandwidth(
            preprocessed,
            sampling_rate=fs,
        )

        snr_res = estimate_snr(
            preprocessed,
            method="difference",
            sampling_rate=fs,
        )

        freq_res = estimate_frequency(
            preprocessed,
            sampling_rate=fs,
        )

        est_freq = float(
            freq_res.get(
                "estimated_frequency_hz",
                0.0,
            )
        )

        cfo_res = estimate_cfo(
            preprocessed,
            sampling_rate=fs,
            reference_frequency_hz=0.0,
            modulation=pred_class,
        )

        fsk_carrier = None

        if pred_class == "FSK":
            fsk_carrier = estimate_fsk_carrier_center(
                preprocessed,
                sampling_rate=fs,
                reference_frequency_hz=0.0,
            )

        phase_freq = abs(est_freq) if abs(est_freq) > 0.0 else None

        phase_res = estimate_phase(
            preprocessed,
            sampling_rate=fs,
            frequency_hz=phase_freq,
        )

        timing_res = estimate_symbol_rate(
            preprocessed,
            sampling_rate=fs,
        )

        constellation_res = analyze_constellation(preprocessed)
        hoc_res = calculate_hoc(preprocessed)

        # Final CFO
        cfo_val = float(
            cfo_res.get(
                "cfo_hz",
                0.0,
            )
        )

        if pred_class == "FSK" and fsk_carrier is not None:
            cfo_val = float(
                fsk_carrier.get(
                    "cfo_hz",
                    cfo_val,
                )
            )

        # =================================================================
        # 7. METADATA REFERENCE
        # =================================================================

        meta_symbol_rate = None

        if isinstance(file_path_or_samples, (str, Path)):
            try:
                paths = resolve_sample_paths(file_path_or_samples)
                meta_path = paths.get("metadata_path")

                if meta_path and meta_path.exists():
                    import json

                    with open(
                        meta_path,
                        "r",
                        encoding="utf-8",
                    ) as f:
                        meta_dict = json.load(f)

                    value = meta_dict.get("symbol_rate_hz")

                    if value:
                        meta_symbol_rate = float(value)

            except Exception:
                meta_symbol_rate = None

        measured_rate = float(
            timing_res["symbol_rate_hz"]
        )

        if meta_symbol_rate is not None:
            discrepancy = (
                abs(measured_rate - meta_symbol_rate)
                / max(meta_symbol_rate, 1.0)
                > 0.05
            )

            timing_status = (
                "Waveform/metadata disagreement"
                if discrepancy
                else timing_res.get(
                    "symbol_rate_status",
                    "detected",
                )
            )
        else:
            discrepancy = False
            timing_status = timing_res.get(
                "symbol_rate_status",
                "detected",
            )

        # =================================================================
        # 8. DSP SUMMARY
        # =================================================================

        dsp_summary = {
            "snr_db": float(
                snr_res.get("snr_db", 0.0)
            ),
            "cfo_hz": cfo_val,
            "cfo_carrier_center_hz": (
                float(fsk_carrier["cfo_hz"])
                if pred_class == "FSK" and fsk_carrier is not None
                else None
            ),
            "fsk_tone_metrics": (
                fsk_carrier
                if pred_class == "FSK" and fsk_carrier is not None
                else None
            ),
            "dominant_positive_tone_hz": float(
                cfo_res.get(
                    "measured_frequency_hz",
                    0.0,
                )
            ),
            "bandwidth_hz": float(
                bw_res.get(
                    "bandwidth_3db_hz",
                    bw_res.get(
                        "bandwidth_hz",
                        0.0,
                    ),
                )
            ),
            "symbol_rate": measured_rate,
            "symbol_rate_waveform": measured_rate,
            "symbol_rate_metadata_ref": meta_symbol_rate,
            "symbol_rate_confidence": float(
                timing_res.get("confidence", 1.0)
            ),
            "symbol_rate_method": timing_res.get(
                "symbol_rate_method",
                "unknown",
            ),
            "symbol_rate_status": timing_status,
            "peak_frequency_hz": float(
                fft_res.get(
                    "peak_frequency_hz",
                    0.0,
                )
            ),
            "phase_offset_rad": float(
                phase_res.get(
                    "phase_offset_radians",
                    0.0,
                )
            ),
            "hoc": hoc_res,
            "constellation": constellation_res,
            "timing_details": timing_res,
        }

        # =================================================================
        # 9. SYNCHRONIZATION
        # =================================================================

        est_cfo = dsp_summary["cfo_hz"]

        est_phase_deg = float(
            phase_res.get(
                "phase_offset_degrees",
                0.0,
            )
        )

        freq_synced = correct_frequency_offset(
            preprocessed,
            est_cfo,
            fs,
        )

        phase_synced = correct_phase_offset(
            freq_synced,
            est_phase_deg,
        )

        try:
            timing_offset = estimate_timing_offset(
                phase_synced,
                samples_per_symbol,
            )

            symbols = sample_symbols(
                phase_synced,
                samples_per_symbol,
                timing_offset,
            )

            sync_status = "SUCCESS"

        except Exception:
            symbols = phase_synced
            timing_offset = 0
            sync_status = "DEGRADED"

        # =================================================================
        # 10. FUSION & EVIDENCE
        # =================================================================

        fused = fuse_dsp_and_ai_features(
            dsp_summary,
            ai_summary,
        )

        evidence = aggregate_evidence(fused)

        # =================================================================
        # 11. DECISION & CONFIDENCE
        # =================================================================

        decision_res = make_modulation_decision(
            pred_class,
            conf_pct,
            evidence,
        )

        composite_conf = compute_composite_confidence(
            conf_pct,
            dsp_summary["snr_db"],
        )

        anomalies = detect_anomalies(
            dsp_summary,
            conf_pct,
        )

        fingerprint = extract_signal_fingerprint(
            preprocessed,
            dsp_summary,
            decision_res["final_modulation"],
        )

        # =================================================================
        # 12. DEMODULATION
        # =================================================================

        final_mod = decision_res["final_modulation"]

        recovered_bits = np.array(
            [],
            dtype=np.uint8,
        )

        try:
            if final_mod == "BPSK":
                recovered_bits = demodulate_bpsk(symbols)

            elif final_mod == "QPSK":
                recovered_bits = demodulate_qpsk(symbols)

            elif final_mod == "FSK":
                recovered_bits = demodulate_fsk(
                    phase_synced,
                    samples_per_symbol,
                )

            elif final_mod == "QAM16":
                recovered_bits = demodulate_16qam(symbols)

            else:
                recovered_bits = demodulate_bpsk(symbols)

        except Exception as e:
            print(
                f"[WARN] Demodulation error for "
                f"{final_mod}: {e}"
            )
            recovered_bits = np.array(
                [],
                dtype=np.uint8,
            )

        # =================================================================
        # 13. DECODING & VALIDATION
        # =================================================================

        decoded_text = None

        if len(recovered_bits) >= 8:
            try:
                decoded_text = bits_to_data(
                    recovered_bits,
                    encoding="ASCII",
                )
            except Exception:
                decoded_text = None

        payload_val = validate_decoded_payload(
            decoded_text or "",
            recovered_bits,
        )

        recovery_val = validate_recovery_pipeline(
            {"status": sync_status},
            {"recovered_bits": recovered_bits},
            payload_val,
        )

        # =================================================================
        # 14. COMPILE REPORT
        # =================================================================

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
                "entropy": payload_val.get(
                    "entropy",
                    0.0,
                ),
                "printable_ratio": payload_val.get(
                    "printable_ratio",
                    0.0,
                ),
            },
            "recovery_validation": recovery_val,
        }

        report = generate_intelligence_report(raw_analysis)
        raw_analysis["report"] = report

        # =================================================================
        # 15. FRONTEND VISUALIZATION DATA
        # =================================================================

        # -----------------------------
        # Waveform
        # -----------------------------

        waveform_pts = 256

        if len(preprocessed) >= waveform_pts:
            step = len(preprocessed) / waveform_pts
            indices = (
                np.arange(waveform_pts) * step
            ).astype(int)

            waveform_samples = (
                preprocessed.real[indices].tolist()
            )

            waveform_time = (
                indices / fs
            ).tolist()

        else:
            waveform_samples = np.pad(
                preprocessed.real,
                (
                    0,
                    waveform_pts - len(preprocessed),
                ),
            ).tolist()

            waveform_time = (
                np.arange(waveform_pts) / fs
            ).tolist()

        # -----------------------------
        # Centered complex-IQ spectrum
        # -----------------------------

        spectrum_pts = 64
        n = len(preprocessed)

        if n > 0:
            iq_fft = np.fft.fftshift(
                np.fft.fft(preprocessed)
            )

            iq_freq = np.fft.fftshift(
                np.fft.fftfreq(n, d=1.0 / fs)
            )

            indices = np.linspace(
                0,
                n - 1,
                spectrum_pts,
            ).astype(int)

            sub_mag = np.abs(iq_fft[indices])

            max_m = float(np.max(sub_mag))
            min_m = float(np.min(sub_mag))

            if max_m > min_m:
                spectrum_bins = (
                    (sub_mag - min_m)
                    / (max_m - min_m)
                ).tolist()
            else:
                spectrum_bins = [0.0] * spectrum_pts

            spectrum_frequencies = (
                iq_freq[indices].tolist()
            )

        else:
            spectrum_bins = [0.0] * spectrum_pts
            spectrum_frequencies = [0.0] * spectrum_pts

        # -----------------------------
        # Spectrogram
        # -----------------------------

        try:
            from ai.representations.spectrogram import (
                compute_spectrogram,
            )
            from scipy.ndimage import zoom

            spec_freq, spec_time, mag_db = (
                compute_spectrogram(
                    preprocessed,
                    sample_rate=fs,
                )
            )

            h, w = 24, 48

            curr_h, curr_w = mag_db.shape

            if curr_h > 0 and curr_w > 0:
                resampled = zoom(
                    mag_db,
                    (
                        h / curr_h,
                        w / curr_w,
                    ),
                    order=1,
                )

                min_db = float(np.min(resampled))
                max_db = float(np.max(resampled))

                if max_db > min_db:
                    spectrogram_rows = (
                        (resampled - min_db)
                        / (max_db - min_db)
                    ).tolist()
                else:
                    spectrogram_rows = [
                        [0.0] * w
                        for _ in range(h)
                    ]

                spectrogram_times = np.linspace(
                    float(spec_time[0]),
                    float(spec_time[-1]),
                    h,
                ).tolist()

                spectrogram_frequencies = np.linspace(
                    float(spec_freq[0]),
                    float(spec_freq[-1]),
                    w,
                ).tolist()

            else:
                raise ValueError("Empty spectrogram")

        except Exception:
            spectrogram_rows = [
                [0.0] * 48
                for _ in range(24)
            ]

            spectrogram_times = (
                np.linspace(
                    0.0,
                    len(preprocessed) / fs,
                    24,
                ).tolist()
            )

            spectrogram_frequencies = (
                np.linspace(
                    -fs / 2,
                    fs / 2,
                    48,
                ).tolist()
            )

        # =================================================================
        # 16. PREDICTION BREAKDOWN
        # =================================================================

        probs = ai_summary.get(
            "probabilities",
            {},
        )

        prediction_breakdown = [
            {
                "classLabel": name,
                "probability": float(
                    probs.get(name, 0.0)
                ) / 100.0,
            }
            for name in CLASS_NAMES
        ]

        prediction_breakdown.sort(
            key=lambda x: x["probability"],
            reverse=True,
        )

        # =================================================================
        # 17. FEATURE TABLE
        # =================================================================

        sr_desc = "Waveform-estimated symbol/baud rate"

        if dsp_summary.get("symbol_rate_metadata_ref") is not None:
            sr_desc += (
                f" (Ref: "
                f"{dsp_summary['symbol_rate_metadata_ref']:.0f} "
                f"Baud | "
                f"{dsp_summary['symbol_rate_status']})"
            )

        if dsp_summary.get("fsk_tone_metrics") is not None:
            fsk_tm = dsp_summary["fsk_tone_metrics"]

            cfo_features = [
                {
                    "name": "Carrier Center Offset",
                    "value": f"{dsp_summary['cfo_hz']:.1f}",
                    "unit": "Hz",
                    "description":
                        "Estimated carrier center offset "
                        "from baseband reference "
                        "(f_center = (f_upper + f_lower)/2)",
                },
                {
                    "name": "Lower FSK Tone",
                    "value": (
                        f"{fsk_tm['dominant_lower_tone_hz'] / 1e3:.1f}"
                    ),
                    "unit": "kHz",
                    "description":
                        "Dominant negative-frequency data tone (f0)",
                },
                {
                    "name": "Upper FSK Tone",
                    "value": (
                        f"{fsk_tm['dominant_upper_tone_hz'] / 1e3:.1f}"
                    ),
                    "unit": "kHz",
                    "description":
                        "Dominant positive-frequency data tone (f1)",
                },
                {
                    "name": "Tone Separation",
                    "value": (
                        f"{fsk_tm['tone_separation_hz'] / 1e3:.1f}"
                    ),
                    "unit": "kHz",
                    "description":
                        "Peak-to-peak tone frequency separation",
                },
            ]

        else:
            cfo_features = [
                {
                    "name": "Carrier Frequency Offset",
                    "value": f"{dsp_summary['cfo_hz']:.1f}",
                    "unit": "Hz",
                    "description":
                        "Estimated carrier frequency shift "
                        "from baseband center",
                },
            ]

        features_list = cfo_features + [
            {
                "name": "Signal-to-Noise Ratio",
                "value": f"{dsp_summary['snr_db']:.1f}",
                "unit": "dB",
                "description":
                    "Estimated signal power over noise floor",
            },
            {
                "name": "Occupied Bandwidth (99% Power)",
                "value": (
                    f"{dsp_summary['bandwidth_hz'] / 1e3:.1f}"
                ),
                "unit": "kHz",
                "description":
                    "Estimated 99% occupied spectral power bandwidth",
            },
            {
                "name": "Symbol Rate",
                "value": f"{dsp_summary['symbol_rate']:.0f}",
                "unit": "Baud",
                "description": sr_desc,
            },
            {
                "name": "Higher-Order Cumulant C40",
                "value": (
                    f"{abs(dsp_summary['hoc'].get('C40', 0)):.3f}"
                ),
                "unit": "",
                "description":
                    "4th-order cumulant reflecting constellation symmetry",
            },
            {
                "name": "Higher-Order Cumulant C42",
                "value": (
                    f"{abs(dsp_summary['hoc'].get('C42', 0)):.3f}"
                ),
                "unit": "",
                "description":
                    "4th-order cumulant reflecting power variance",
            },
            {
                "name": "Peak-to-Average Power Ratio",
                "value": (
                    f"{fingerprint.get('papr_db', 0):.1f}"
                ),
                "unit": "dB",
                "description":
                    "Crest factor of signal envelope",
            },
            {
                "name": "Emitter RF Fingerprint ID",
                "value": (
                    f"{fingerprint.get('fingerprint_id', 'N/A')}"
                ),
                "unit": "",
                "description":
                    "Derived RF feature-hash identifier",
            },
        ]

        # =================================================================
        # 18. EXPLANATION
        # =================================================================

        if dsp_summary.get("fsk_tone_metrics") is not None:
            fsk_tm = dsp_summary["fsk_tone_metrics"]

            cfo_exp_str = (
                f"carrier center offset of "
                f"{dsp_summary['cfo_hz']:.1f} Hz "
                f"(tones: "
                f"{fsk_tm['dominant_lower_tone_hz'] / 1e3:.1f} / "
                f"{fsk_tm['dominant_upper_tone_hz'] / 1e3:.1f} kHz)"
            )

        else:
            cfo_exp_str = (
                f"carrier offset of "
                f"{dsp_summary['cfo_hz']:.1f} Hz"
            )

        explanation = (
            f"The AI Transformer model classified this signal as "
            f"{final_mod} with {conf_pct:.1f}% confidence "
            f"({det_status}). "

            f"DSP physical analysis provides supporting evidence "
            f"with an estimated SNR of "
            f"{dsp_summary['snr_db']:.1f} dB, "

            f"{cfo_exp_str}, "

            f"and occupied bandwidth of "
            f"{dsp_summary['bandwidth_hz'] / 1e3:.1f} kHz "
            f"(99% power). "

            f"Higher-Order Cumulants "
            f"(C40={abs(dsp_summary['hoc'].get('C40', 0)):.3f}, "
            f"C42={abs(dsp_summary['hoc'].get('C42', 0)):.3f}) "

            f"and spectral profile support this hypothesis "
            f"with multi-modal evidence score of "
            f"{evidence.get('overall_evidence_score', 1.0):.2f}. "

            f"Demodulation produced "
            f"{len(recovered_bits)} recovered raw bits; "
            f"data validation has not yet been established."
        )

        # =================================================================
        # 19. FRONTEND DATA
        # =================================================================

        frontend_data = {
            "id": f"analysis-{sample_id}",
            "timestamp": (
                datetime.datetime.utcnow().isoformat()
                + "Z"
            ),
            "isDemoData": False,
            "label": f"Live SYNAPS Analysis ({final_mod})",
            "filename": (
                Path(file_str).name
                if file_str != "in_memory"
                else "in_memory_signal.iq"
            ),
            "format": fmt,
            "classification": final_mod,
            "confidence": float(conf_pct) / 100.0,
            "sampleRate": fs,
            "duration": len(raw_samples) / fs,
            "bandwidth": dsp_summary["bandwidth_hz"],
            "snr": dsp_summary["snr_db"],
            "peakFrequency": dsp_summary["peak_frequency_hz"],
            "numSamples": len(raw_samples),
            "predictionBreakdown": prediction_breakdown,
            "features": features_list,
            "explanation": explanation,

            # Existing visualization fields
            "waveformSamples": waveform_samples,
            "spectrumBins": spectrum_bins,
            "spectrogramRows": spectrogram_rows,

            # New physical axes
            "waveformTime": waveform_time,
            "spectrumFrequencies": spectrum_frequencies,
            "spectrogramTimes": spectrogram_times,
            "spectrogramFrequencies": spectrogram_frequencies,

            "raw_report": report,
        }

        raw_analysis["frontend_data"] = frontend_data

        return raw_analysis


# ============================================================================
# GLOBAL DEFAULT SERVICE INSTANCE
# ============================================================================

default_pipeline = AnalysisPipeline()


def analyze_signal(
    file_path_or_samples,
    sample_rate=None,
    samples_per_symbol=10,
    **kwargs,
):
    """Convenience function to analyze any signal via the default pipeline."""

    return default_pipeline.run(
        file_path_or_samples,
        sample_rate=sample_rate,
        samples_per_symbol=samples_per_symbol,
        **kwargs,
    )