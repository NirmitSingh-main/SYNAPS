"""
Complete End-to-End Blind Synchronization and Symbol Extraction Pipeline.

Pipeline:
    Raw IQ
      ↓
    RMS Normalization
      ↓
    Blind CFO Estimation & Derotation
      ↓
    Blind Symbol-Rate / SPS Estimation
      ↓
    Feedforward Fractional Timing Recovery (Oerder-Meyr)
      ↓
    Symbol-Center Constellation Samples
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np

from dsp.frequency.cfo import estimate_cfo
from dsp.timing.symbol_rate import estimate_symbol_rate
from .frequency_sync import correct_frequency_offset
from .timing_sync import estimate_timing_offset, sample_symbols


@dataclass
class SynchronizationResult:
    """Result of blind synchronization pipeline."""
    symbols: np.ndarray
    cfo_hz: float
    symbol_rate_hz: float
    samples_per_symbol: float
    timing_offset_samples: float
    cfo_corrected_iq: np.ndarray
    raw_iq_normalized: np.ndarray


def extract_synchronized_symbols(
    iq: np.ndarray,
    sampling_rate: float = 1_000_000.0,
    modulation_hint: Optional[str] = None,
    use_fractional_timing: bool = True,
) -> SynchronizationResult:
    """
    Perform complete blind carrier and timing synchronization to extract
    symbol-center constellation samples.

    Parameters:
    -----------
    iq : np.ndarray
        Raw complex IQ samples.
    sampling_rate : float
        Sampling frequency in Hz (default: 1 MHz).
    modulation_hint : Optional[str]
        Optional modulation hint if available (default: None for fully blind).
    use_fractional_timing : bool
        Whether to use continuous fractional timing interpolation (default: True).

    Returns:
    --------
    SynchronizationResult
        Dataclass containing symbol samples, estimated CFO, SPS, timing offset,
        and intermediate signals.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    if iq.size == 0:
        raise ValueError("Input IQ signal cannot be empty.")

    # 1. Safe RMS power normalization
    power = np.mean(np.abs(iq) ** 2)
    scale = np.sqrt(power) if power > 1e-12 else 1.0
    norm_iq = iq / scale

    # 2. Blind Carrier Frequency Offset (CFO) Estimation
    cfo_dict = estimate_cfo(
        norm_iq,
        sampling_rate,
        reference_frequency_hz=0.0,
        modulation=modulation_hint,
    )
    cfo_hz = float(cfo_dict["cfo_hz"])

    # 3. CFO Derotation
    cfo_corrected_iq = correct_frequency_offset(
        norm_iq,
        cfo_hz,
        sampling_rate,
    )

    # 4. Symbol Rate / Samples-per-Symbol Estimation
    try:
        sr_dict = estimate_symbol_rate(
            cfo_corrected_iq,
            sampling_rate,
            minimum_symbol_rate_hz=20_000.0,
            maximum_symbol_rate_hz=250_000.0,
        )
        sps = float(sr_dict["samples_per_symbol"])
        symbol_rate_hz = float(sr_dict["symbol_rate_hz"])
    except Exception:
        # Fallback to nominal 10 SPS
        sps = 10.0
        symbol_rate_hz = sampling_rate / 10.0

    # 5. Feedforward Timing Phase Recovery
    timing_offset = estimate_timing_offset(
        cfo_corrected_iq,
        sps,
        return_fractional=use_fractional_timing,
    )

    # 6. Extract Symbol-Spaced Constellation Samples
    symbols = sample_symbols(
        cfo_corrected_iq,
        sps,
        timing_offset=timing_offset,
        use_fractional=use_fractional_timing,
    )

    # Ensure symbol power is normalized
    sym_pwr = np.mean(np.abs(symbols) ** 2)
    if sym_pwr > 1e-12:
        symbols = symbols / np.sqrt(sym_pwr)

    return SynchronizationResult(
        symbols=symbols,
        cfo_hz=cfo_hz,
        symbol_rate_hz=symbol_rate_hz,
        samples_per_symbol=sps,
        timing_offset_samples=float(timing_offset),
        cfo_corrected_iq=cfo_corrected_iq,
        raw_iq_normalized=norm_iq,
    )
