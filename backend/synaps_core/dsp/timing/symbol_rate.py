"""Waveform-derived symbol-rate estimation utilities."""

from math import gcd
from typing import Any, Optional

import numpy as np
from scipy.signal import find_peaks, medfilt


def _validate_signal(signal: np.ndarray, sampling_rate: float) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")
    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0.0:
        raise ValueError("sampling_rate must be greater than zero.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Signal contains NaN or infinite values.")
    return signal.astype(np.complex128 if np.iscomplexobj(signal) else np.float64)


def _prepare_timing_signal(signal: np.ndarray) -> np.ndarray:
    timing_signal = np.abs(signal) ** 2 if np.iscomplexobj(signal) else np.asarray(signal, dtype=np.float64) ** 2
    return np.asarray(timing_signal, dtype=np.float64) - np.mean(timing_signal)


def _calculate_timing_spectrum(timing_signal: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    if timing_signal.size < 4:
        raise ValueError("At least four samples are required for symbol-rate estimation.")
    spectrum = np.fft.rfft(timing_signal * np.hanning(timing_signal.size))
    frequencies = np.fft.rfftfreq(timing_signal.size, d=1.0 / sampling_rate)
    magnitude = np.abs(spectrum)
    magnitude[0] = 0.0
    return frequencies, magnitude


def _find_symbol_rate_candidates(
    frequencies: np.ndarray,
    magnitude: np.ndarray,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    maximum_candidates: int = 10,
) -> list[tuple[float, float]]:
    mask = (frequencies >= minimum_frequency_hz) & (frequencies <= maximum_frequency_hz)
    if not np.any(mask):
        return []
    candidate_frequencies = frequencies[mask]
    candidate_magnitude = magnitude[mask]
    if candidate_magnitude.size == 0 or np.max(candidate_magnitude) <= 0.0:
        return []
    maximum_magnitude = float(np.max(candidate_magnitude))
    peaks, properties = find_peaks(
        candidate_magnitude,
        height=maximum_magnitude * 0.05,
        distance=max(1, candidate_magnitude.size // 200),
    )
    if peaks.size == 0:
        strongest = int(np.argmax(candidate_magnitude))
        return [(float(candidate_frequencies[strongest]), float(candidate_magnitude[strongest]))]
    heights = properties.get("peak_heights", candidate_magnitude[peaks])
    order = np.argsort(heights)[::-1]
    return [
        (float(candidate_frequencies[int(peaks[index])]), float(heights[index]))
        for index in order[:maximum_candidates]
    ]


def _select_fundamental(
    candidates: list[tuple[float, float]],
    magnitude: np.ndarray,
    frequencies: np.ndarray,
    tolerance: float = 0.03,
) -> tuple[float, float]:
    if not candidates:
        raise ValueError("Unable to find a symbol-rate candidate.")
    maximum_strength = max(strength for _, strength in candidates)
    if maximum_strength <= 0.0:
        return candidates[0][0], 0.0
    scored = []
    for rate, strength in candidates:
        score = strength / maximum_strength
        for divisor in (2, 3, 4):
            possible = rate / divisor
            nearest = int(np.argmin(np.abs(frequencies - possible)))
            relative_error = abs(frequencies[nearest] - possible) / possible
            if relative_error <= tolerance and magnitude[nearest] > 0.10 * strength:
                score = max(score, 0.85 * float(magnitude[nearest]) / maximum_strength)
        scored.append((rate, score))
    scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return float(scored[0][0]), float(np.clip(scored[0][1], 0.0, 1.0))


def _estimate_autocorrelation_candidate(
    signal: np.ndarray,
    sampling_rate: float,
    minimum_symbol_rate_hz: float,
    maximum_symbol_rate_hz: float,
) -> Optional[tuple[float, float, float]]:
    centered = signal - np.mean(signal)
    number_of_samples = centered.size
    maximum_lag = min(number_of_samples // 4, int(np.ceil(sampling_rate / minimum_symbol_rate_hz * 2.0)))
    if maximum_lag < 3:
        return None
    lags = np.arange(1, maximum_lag + 1, dtype=np.int64)
    autocorrelation = np.array([
        abs(np.vdot(centered[:-lag], centered[lag:])) / (number_of_samples - lag)
        for lag in lags
    ])
    if autocorrelation[0] <= 0.0:
        return None
    normalized = autocorrelation / autocorrelation[0]
    crossings = np.flatnonzero(normalized <= 0.05)
    if crossings.size == 0:
        return None
    crossing_index = int(crossings[0])
    fit_start = max(0, crossing_index - 4)
    fit_lags = lags[fit_start:crossing_index + 1].astype(np.float64)
    fit_values = normalized[fit_start:crossing_index + 1]
    if fit_lags.size < 2:
        return None
    slope, intercept = np.polyfit(fit_lags, fit_values, 1)
    if slope >= 0.0:
        return None
    samples_per_symbol = -intercept / slope
    symbol_rate = sampling_rate / samples_per_symbol
    if not np.isfinite(symbol_rate) or not minimum_symbol_rate_hz <= symbol_rate <= maximum_symbol_rate_hz:
        return None
    residual = float(np.mean((fit_values - (slope * fit_lags + intercept)) ** 2))
    confidence = float(np.clip(1.0 - residual / 0.05, 0.0, 1.0))
    return float(symbol_rate), confidence, float(samples_per_symbol)


def _estimate_fsk_candidate(
    signal: np.ndarray,
    sampling_rate: float,
    minimum_symbol_rate_hz: float,
    maximum_symbol_rate_hz: float,
) -> Optional[tuple[float, float, float]]:
    if signal.size < 16 or not np.iscomplexobj(signal):
        return None
    phase_difference = np.angle(signal[1:] * np.conjugate(signal[:-1]))
    absolute_phase_difference = np.abs(phase_difference)
    percentiles = np.percentile(absolute_phase_difference, [50, 95, 99])
    if percentiles[1] > 0.8 or percentiles[1] - percentiles[0] > 0.3:
        return None
    phase_difference = medfilt(phase_difference, kernel_size=5)
    states = phase_difference > float(np.median(phase_difference))
    changes = np.flatnonzero(np.diff(states.astype(np.int8)) != 0) + 1
    run_lengths = np.diff(np.r_[0, changes, phase_difference.size])
    run_lengths = run_lengths[run_lengths >= 3]
    if run_lengths.size < 4:
        return None
    values, counts = np.unique(run_lengths, return_counts=True)
    samples_per_symbol = float(values[int(np.argmax(counts))])
    symbol_rate = sampling_rate / samples_per_symbol
    if not minimum_symbol_rate_hz <= symbol_rate <= maximum_symbol_rate_hz:
        return None
    spread = float(np.median(np.abs(run_lengths - samples_per_symbol)))
    confidence = float(np.clip(1.0 - spread / max(samples_per_symbol, 1.0), 0.0, 1.0))
    return float(symbol_rate), confidence, samples_per_symbol


def _estimate_transition_candidate(
    signal: np.ndarray,
    sampling_rate: float,
    minimum_symbol_rate_hz: float,
    maximum_symbol_rate_hz: float,
) -> Optional[tuple[float, float, float]]:
    """Estimate rectangular-symbol timing from transition periodicity."""
    if signal.size < 32:
        return None
    transitions = np.abs(np.diff(signal)) ** 2
    transitions -= np.mean(transitions)
    maximum_lag = min(signal.size // 4, int(np.ceil(sampling_rate / minimum_symbol_rate_hz * 2.0)))
    if maximum_lag < 4:
        return None
    autocorrelation = np.array([
        abs(np.vdot(transitions[:-lag], transitions[lag:])) / (transitions.size - lag)
        for lag in range(1, maximum_lag + 1)
    ])
    if autocorrelation.size == 0 or np.max(autocorrelation) <= 0.0:
        return None
    peak_level = 0.55 * float(np.max(autocorrelation))
    peak_lags = [
        index + 1
        for index in range(1, autocorrelation.size - 1)
        if autocorrelation[index] >= autocorrelation[index - 1]
        and autocorrelation[index] >= autocorrelation[index + 1]
        and autocorrelation[index] >= peak_level
    ]
    if len(peak_lags) < 2:
        return None
    samples_per_symbol = 0
    for peak_lag in peak_lags[:12]:
        samples_per_symbol = gcd(samples_per_symbol, peak_lag)
    if samples_per_symbol < 2:
        return None
    symbol_rate = sampling_rate / samples_per_symbol
    if not minimum_symbol_rate_hz <= symbol_rate <= maximum_symbol_rate_hz:
        return None
    confidence = float(np.clip(len(peak_lags) / 8.0, 0.0, 1.0))
    return float(symbol_rate), confidence, float(samples_per_symbol)


def estimate_symbol_rate(
    signal: np.ndarray,
    sampling_rate: float,
    minimum_symbol_rate_hz: float = 1.0,
    maximum_symbol_rate_hz: Optional[float] = None,
) -> dict[str, Any]:
    """Estimate symbol rate and expose candidate/method diagnostics."""
    signal = _validate_signal(signal, sampling_rate)
    if not np.isfinite(minimum_symbol_rate_hz) or minimum_symbol_rate_hz <= 0.0:
        raise ValueError("minimum_symbol_rate_hz must be greater than zero.")
    nyquist_frequency = sampling_rate / 2.0
    if maximum_symbol_rate_hz is None:
        maximum_symbol_rate_hz = nyquist_frequency
    elif not np.isfinite(maximum_symbol_rate_hz) or maximum_symbol_rate_hz <= 0.0:
        raise ValueError("maximum_symbol_rate_hz must be greater than zero.")
    maximum_symbol_rate_hz = min(float(maximum_symbol_rate_hz), nyquist_frequency)
    if minimum_symbol_rate_hz >= maximum_symbol_rate_hz:
        raise ValueError("minimum_symbol_rate_hz must be less than maximum_symbol_rate_hz.")

    timing_signal = _prepare_timing_signal(signal)
    frequencies, magnitude = _calculate_timing_spectrum(timing_signal, sampling_rate)
    frequency_resolution = float(sampling_rate / signal.size)
    spectral_candidates = _find_symbol_rate_candidates(
        frequencies, magnitude, float(minimum_symbol_rate_hz), float(maximum_symbol_rate_hz)
    )
    specialized_candidate = _estimate_fsk_candidate(
        signal, sampling_rate, float(minimum_symbol_rate_hz), float(maximum_symbol_rate_hz)
    )
    method = "instantaneous_frequency_state_duration"
    if specialized_candidate is None:
        specialized_candidate = _estimate_transition_candidate(
            signal, sampling_rate, float(minimum_symbol_rate_hz), float(maximum_symbol_rate_hz)
        )
        method = "transition_energy_periodicity"
    if specialized_candidate is None:
        specialized_candidate = _estimate_autocorrelation_candidate(
            signal, sampling_rate, float(minimum_symbol_rate_hz), float(maximum_symbol_rate_hz)
        )
        method = "complex_autocorrelation_decay"

    if specialized_candidate is not None:
        symbol_rate, confidence, samples_per_symbol = specialized_candidate
    elif spectral_candidates:
        symbol_rate, confidence = _select_fundamental(spectral_candidates, magnitude, frequencies)
        samples_per_symbol = sampling_rate / symbol_rate
        method = "magnitude_squared_fft"
    else:
        raise ValueError("Unable to detect a symbol-rate candidate.")

    candidate_rates = [float(rate) for rate, _ in spectral_candidates]
    candidate_scores = [float(strength) for _, strength in spectral_candidates]
    if specialized_candidate is not None:
        candidate_rates.insert(0, float(symbol_rate))
        candidate_scores.insert(0, float(confidence))
    nearest_index = int(np.argmin(np.abs(frequencies - symbol_rate)))
    status = "detected" if confidence >= 0.5 else "low_confidence"

    return {
        "symbol_rate_hz": float(symbol_rate),
        "samples_per_symbol": float(samples_per_symbol),
        "confidence": float(confidence),
        "symbol_rate_confidence": float(confidence),
        "symbol_rate_error_estimate": frequency_resolution,
        "symbol_rate_method": method,
        "symbol_rate_status": status,
        "candidate_rates": candidate_rates,
        "candidate_scores": candidate_scores,
        "timing_peak_frequency_hz": float(frequencies[nearest_index]),
        "timing_peak_magnitude": float(magnitude[nearest_index]),
        "frequency_resolution_hz": frequency_resolution,
        "number_of_samples": int(signal.size),
        "sampling_rate_hz": float(sampling_rate),
        "is_complex": bool(np.iscomplexobj(signal)),
    }


def estimate_symbol_rate_from_signal(
    signal: np.ndarray,
    sampling_rate: float,
    minimum_symbol_rate_hz: float = 1.0,
    maximum_symbol_rate_hz: Optional[float] = None,
) -> dict[str, Any]:
    return estimate_symbol_rate(signal, sampling_rate, minimum_symbol_rate_hz, maximum_symbol_rate_hz)


def get_symbol_rate(
    signal: np.ndarray,
    sampling_rate: float,
    minimum_symbol_rate_hz: float = 1.0,
    maximum_symbol_rate_hz: Optional[float] = None,
) -> float:
    return float(estimate_symbol_rate(signal, sampling_rate, minimum_symbol_rate_hz, maximum_symbol_rate_hz)["symbol_rate_hz"])
