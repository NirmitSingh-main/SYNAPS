# """
# Constellation analysis for communication signals.

# Supported inputs:
#     - Real-valued signals
#     - Complex In-phase/Quadrature signals
#     - WAV files
#     - Raw IQ files

# This module extracts constellation-related Digital Signal
# Processing features.

# It does not directly decide the modulation type.
# The extracted features can later be supplied to the
# fusion and intelligence layers.
# """

# from pathlib import Path
# from typing import Any, Optional, Tuple

# import numpy as np
# from scipy.io import wavfile
# from scipy.signal import hilbert


# # ---------------------------------------------------------------------
# # SIGNAL VALIDATION
# # ---------------------------------------------------------------------

# def _validate_signal(
#     signal: np.ndarray,
#     sampling_rate: float,
# ) -> np.ndarray:
#     """
#     Validate a one-dimensional signal.
#     """

#     signal = np.asarray(signal)

#     if signal.size == 0:
#         raise ValueError(
#             "Signal cannot be empty."
#         )

#     if signal.ndim != 1:
#         raise ValueError(
#             "Signal must be one-dimensional."
#         )

#     if sampling_rate <= 0:
#         raise ValueError(
#             "Sampling rate must be greater than zero."
#         )

#     if not np.all(np.isfinite(signal)):
#         raise ValueError(
#             "Signal contains NaN or infinite values."
#         )

#     return signal


# # ---------------------------------------------------------------------
# # SIGNAL PREPARATION
# # ---------------------------------------------------------------------

# def _prepare_signal(
#     signal: np.ndarray,
# ) -> np.ndarray:
#     """
#     Convert the input signal into a complex signal.

#     Complex In-phase/Quadrature signals are used directly.

#     Real-valued signals are converted into an analytic signal
#     using the Hilbert transform.
#     """

#     signal = np.asarray(signal)

#     if np.iscomplexobj(signal):
#         return np.asarray(
#             signal,
#             dtype=np.complex128,
#         )

#     real_signal = np.asarray(
#         signal,
#         dtype=np.float64,
#     )

#     analytic_signal = hilbert(
#         real_signal
#     )

#     return np.asarray(
#         analytic_signal,
#         dtype=np.complex128,
#     )


# # ---------------------------------------------------------------------
# # NORMALIZATION
# # ---------------------------------------------------------------------

# def _normalize_signal(
#     signal: np.ndarray,
# ) -> np.ndarray:
#     """
#     Normalize the signal using root-mean-square amplitude.
#     """

#     signal = np.asarray(
#         signal,
#         dtype=np.complex128,
#     )

#     rms = float(
#         np.sqrt(
#             np.mean(
#                 np.abs(signal) ** 2
#             )
#         )
#     )

#     if rms == 0.0:
#         raise ValueError(
#             "Cannot normalize a zero-energy signal."
#         )

#     return signal / rms


# # ---------------------------------------------------------------------
# # CONSTELLATION POINT PREPARATION
# # ---------------------------------------------------------------------

# def get_constellation_points(
#     signal: np.ndarray,
#     normalize: bool = True,
# ) -> np.ndarray:
#     """
#     Convert a real or complex signal into two-dimensional
#     constellation points.

#     Returns:
#         NumPy array with shape:

#             (number_of_samples, 2)

#         Column 0 = In-phase
#         Column 1 = Quadrature
#     """

#     complex_signal = _prepare_signal(
#         signal
#     )

#     if normalize:
#         complex_signal = _normalize_signal(
#             complex_signal
#         )

#     points = np.column_stack(
#         (
#             np.real(complex_signal),
#             np.imag(complex_signal),
#         )
#     )

#     return np.asarray(
#         points,
#         dtype=np.float64,
#     )


# # ---------------------------------------------------------------------
# # BASIC CONSTELLATION STATISTICS
# # ---------------------------------------------------------------------

# def _calculate_statistics(
#     signal: np.ndarray,
# ) -> dict[str, float]:
#     """
#     Calculate basic constellation statistics.
#     """

#     i_samples = np.real(signal)
#     q_samples = np.imag(signal)

#     amplitude = np.abs(signal)

#     phase = np.angle(signal)

#     # Circular mean phase.
#     mean_phase = np.angle(
#         np.mean(
#             np.exp(1j * phase)
#         )
#     )

#     return {
#         "mean_i": float(
#             np.mean(i_samples)
#         ),
#         "mean_q": float(
#             np.mean(q_samples)
#         ),
#         "std_i": float(
#             np.std(i_samples)
#         ),
#         "std_q": float(
#             np.std(q_samples)
#         ),
#         "mean_amplitude": float(
#             np.mean(amplitude)
#         ),
#         "std_amplitude": float(
#             np.std(amplitude)
#         ),
#         "minimum_amplitude": float(
#             np.min(amplitude)
#         ),
#         "maximum_amplitude": float(
#             np.max(amplitude)
#         ),
#         "mean_phase_radians": float(
#             mean_phase
#         ),
#         "phase_std_radians": float(
#             np.std(phase)
#         ),
#     }


# # ---------------------------------------------------------------------
# # CONSTELLATION GEOMETRY
# # ---------------------------------------------------------------------

# def _calculate_geometry(
#     signal: np.ndarray,
# ) -> dict[str, float]:
#     """
#     Calculate geometric properties of the constellation.
#     """

#     i_samples = np.real(signal)
#     q_samples = np.imag(signal)

#     mean_i = float(
#         np.mean(i_samples)
#     )

#     mean_q = float(
#         np.mean(q_samples)
#     )

#     distance_from_center = np.sqrt(
#         (
#             i_samples - mean_i
#         ) ** 2
#         +
#         (
#             q_samples - mean_q
#         ) ** 2
#     )

#     mean_distance = float(
#         np.mean(distance_from_center)
#     )

#     distance_std = float(
#         np.std(distance_from_center)
#     )

#     return {
#         "constellation_center_i": mean_i,
#         "constellation_center_q": mean_q,
#         "mean_distance_from_center": mean_distance,
#         "distance_from_center_std": distance_std,
#     }


# # ---------------------------------------------------------------------
# # RADIAL AND PHASE CHARACTERISTICS
# # ---------------------------------------------------------------------

# def _calculate_radial_statistics(
#     signal: np.ndarray,
# ) -> dict[str, float]:
#     """
#     Calculate radial and angular characteristics.
#     """

#     amplitude = np.abs(signal)
#     phase = np.angle(signal)

#     amplitude_mean = float(
#         np.mean(amplitude)
#     )

#     if amplitude_mean == 0.0:
#         amplitude_coefficient_of_variation = 0.0
#     else:
#         amplitude_coefficient_of_variation = (
#             float(np.std(amplitude))
#             / amplitude_mean
#         )

#     # Circular concentration.
#     phase_concentration = float(
#         np.abs(
#             np.mean(
#                 np.exp(1j * phase)
#             )
#         )
#     )

#     return {
#         "amplitude_coefficient_of_variation": (
#             amplitude_coefficient_of_variation
#         ),
#         "phase_concentration": (
#             phase_concentration
#         ),
#     }


# # ---------------------------------------------------------------------
# # CONSTELLATION SYMMETRY
# # ---------------------------------------------------------------------

# def _calculate_symmetry(
#     signal: np.ndarray,
# ) -> dict[str, float]:
#     """
#     Estimate rotational symmetry of the constellation.

#     This does not classify modulation. It simply provides
#     useful features for later intelligence processing.
#     """

#     signal = np.asarray(
#         signal,
#         dtype=np.complex128,
#     )

#     energy = float(
#         np.mean(
#             np.abs(signal) ** 2
#         )
#     )

#     if energy == 0.0:
#         return {
#             "symmetry_90_degree_error": 0.0,
#             "symmetry_180_degree_error": 0.0,
#             "symmetry_270_degree_error": 0.0,
#         }

#     rotated_90 = signal * np.exp(
#         1j * np.pi / 2.0
#     )

#     rotated_180 = signal * np.exp(
#         1j * np.pi
#     )

#     rotated_270 = signal * np.exp(
#         1j * 3.0 * np.pi / 2.0
#     )

#     def normalized_error(
#         original: np.ndarray,
#         rotated: np.ndarray,
#     ) -> float:

#         error = np.mean(
#             np.abs(
#                 original - rotated
#             ) ** 2
#         )

#         return float(
#             error / energy
#         )

#     return {
#         "symmetry_90_degree_error": (
#             normalized_error(
#                 signal,
#                 rotated_90,
#             )
#         ),
#         "symmetry_180_degree_error": (
#             normalized_error(
#                 signal,
#                 rotated_180,
#             )
#         ),
#         "symmetry_270_degree_error": (
#             normalized_error(
#                 signal,
#                 rotated_270,
#             )
#         ),
#     }


# # ---------------------------------------------------------------------
# # MAIN CONSTELLATION ANALYSIS
# # ---------------------------------------------------------------------

# def analyze_constellation(
#     signal: np.ndarray,
#     sampling_rate: float,
#     normalize: bool = True,
# ) -> dict[str, Any]:
#     """
#     Analyze the constellation of a signal.

#     Parameters
#     ----------
#     signal:
#         One-dimensional real or complex signal.

#     sampling_rate:
#         Sampling rate in Hertz.

#     normalize:
#         Normalize signal amplitude before analysis.

#     Returns
#     -------
#     dict
#         Dictionary containing constellation features.
#     """

#     signal = _validate_signal(
#         signal,
#         sampling_rate,
#     )

#     complex_signal = _prepare_signal(
#         signal
#     )

#     if normalize:
#         complex_signal = _normalize_signal(
#             complex_signal
#         )

#     statistics = _calculate_statistics(
#         complex_signal
#     )

#     geometry = _calculate_geometry(
#         complex_signal
#     )

#     radial_statistics = (
#         _calculate_radial_statistics(
#             complex_signal
#         )
#     )

#     symmetry = _calculate_symmetry(
#         complex_signal
#     )

#     result: dict[str, Any] = {
#         "number_of_samples": int(
#             complex_signal.size
#         ),
#         "sampling_rate_hz": float(
#             sampling_rate
#         ),
#         "normalized": bool(
#             normalize
#         ),
#     }

#     result.update(
#         statistics
#     )

#     result.update(
#         geometry
#     )

#     result.update(
#         radial_statistics
#     )

#     result.update(
#         symmetry
#     )

#     return result


# # ---------------------------------------------------------------------
# # WAV FILE LOADING
# # ---------------------------------------------------------------------

# def _load_wav_file(
#     file_path: Path,
# ) -> Tuple[np.ndarray, float]:
#     """
#     Load a WAV file.

#     Stereo WAV files are converted to mono.
#     """

#     sampling_rate, signal = wavfile.read(
#         file_path
#     )

#     signal = np.asarray(signal)

#     if signal.ndim == 2:

#         signal = np.mean(
#             signal.astype(
#                 np.float64
#             ),
#             axis=1,
#         )

#     if np.issubdtype(
#         signal.dtype,
#         np.integer,
#     ):

#         signal = signal.astype(
#             np.float64
#         )

#         maximum_amplitude = float(
#             np.max(
#                 np.abs(signal)
#             )
#         )

#         if maximum_amplitude > 0.0:

#             signal = (
#                 signal
#                 / maximum_amplitude
#             )

#     else:

#         signal = signal.astype(
#             np.float64
#         )

#     return (
#         signal,
#         float(sampling_rate),
#     )


# # ---------------------------------------------------------------------
# # IQ FILE LOADING
# # ---------------------------------------------------------------------

# def _load_iq_file(
#     file_path: Path,
#     iq_dtype: np.dtype = np.dtype(
#         np.float32
#     ),
#     iq_order: str = "IQ",
# ) -> np.ndarray:
#     """
#     Load a raw interleaved IQ file.

#     Expected format:

#         I, Q, I, Q, I, Q, ...

#     iq_order:
#         "IQ" -> I,Q,I,Q,...

#         "QI" -> Q,I,Q,I,...
#     """

#     raw_data = np.fromfile(
#         file_path,
#         dtype=iq_dtype,
#     )

#     if raw_data.size == 0:
#         raise ValueError(
#             "IQ file is empty."
#         )

#     if raw_data.size % 2 != 0:
#         raise ValueError(
#             "IQ file must contain an even number "
#             "of values."
#         )

#     first_component = raw_data[0::2]
#     second_component = raw_data[1::2]

#     order = iq_order.upper()

#     if order == "IQ":

#         i_samples = first_component
#         q_samples = second_component

#     elif order == "QI":

#         q_samples = first_component
#         i_samples = second_component

#     else:

#         raise ValueError(
#             "iq_order must be either 'IQ' or 'QI'."
#         )

#     signal = (
#         i_samples.astype(
#             np.float64
#         )
#         +
#         1j
#         *
#         q_samples.astype(
#             np.float64
#         )
#     )

#     return np.asarray(
#         signal,
#         dtype=np.complex128,
#     )


# # ---------------------------------------------------------------------
# # FILE-BASED CONSTELLATION ANALYSIS
# # ---------------------------------------------------------------------

# def analyze_constellation_from_file(
#     file_path: str | Path,
#     sampling_rate: Optional[float] = None,
#     normalize: bool = True,
#     iq_dtype: np.dtype = np.dtype(
#         np.float32
#     ),
#     iq_order: str = "IQ",
# ) -> dict[str, Any]:
#     """
#     Analyze constellation directly from a WAV or IQ file.

#     WAV:
#         Sampling rate is read from the WAV header.

#     IQ:
#         Sampling rate must be supplied because raw IQ files
#         normally do not contain metadata.
#     """

#     file_path = Path(
#         file_path
#     )

#     if not file_path.exists():

#         raise FileNotFoundError(
#             f"File not found: {file_path}"
#         )

#     extension = file_path.suffix.lower()

#     # -------------------------------------------------------------
#     # WAV
#     # -------------------------------------------------------------

#     if extension == ".wav":

#         signal, file_sampling_rate = (
#             _load_wav_file(
#                 file_path
#             )
#         )

#         result = analyze_constellation(
#             signal,
#             file_sampling_rate,
#             normalize=normalize,
#         )

#         result["sampling_rate_hz"] = float(
#             file_sampling_rate
#         )

#     # -------------------------------------------------------------
#     # IQ
#     # -------------------------------------------------------------

#     elif extension == ".iq":

#         if sampling_rate is None:

#             raise ValueError(
#                 "sampling_rate is required "
#                 "for raw IQ files."
#             )

#         signal = _load_iq_file(
#             file_path,
#             iq_dtype=iq_dtype,
#             iq_order=iq_order,
#         )

#         result = analyze_constellation(
#             signal,
#             sampling_rate,
#             normalize=normalize,
#         )

#         result["sampling_rate_hz"] = float(
#             sampling_rate
#         )

#     # -------------------------------------------------------------
#     # UNSUPPORTED
#     # -------------------------------------------------------------

#     else:

#         raise ValueError(
#             "Unsupported file format. "
#             "Only .wav and .iq files are supported."
#         )

#     result["file"] = str(
#         file_path
#     )

#     result["file_type"] = extension

#     return result


# # ---------------------------------------------------------------------
# # SIMPLE ALIASES
# # ---------------------------------------------------------------------

# def estimate_constellation(
#     signal: np.ndarray,
#     sampling_rate: float,
#     normalize: bool = True,
# ) -> dict[str, Any]:
#     """
#     Alias for analyze_constellation().
#     """

#     return analyze_constellation(
#         signal,
#         sampling_rate,
#         normalize=normalize,
#     )
"""
Constellation analysis for communication signals.

Supported inputs:

    - Real-valued NumPy signals
    - Complex In-phase/Quadrature signals
    - WAV files
    - Raw IQ files

The module:

    1. Validates the signal.
    2. Converts real signals into analytic signals.
    3. Extracts In-phase and Quadrature components.
    4. Removes complex DC offset.
    5. Normalizes the signal.
    6. Extracts constellation points.
    7. Calculates constellation statistics.
    8. Performs constellation clustering.

This module does NOT directly identify the modulation type.

The resulting features are intended for:

    - Machine Learning
    - Feature Fusion
    - Intelligence Engine
    - Signal Fingerprint
"""


from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert


# =====================================================================
# SIGNAL VALIDATION
# =====================================================================

def _validate_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Validate an input signal.
    """

    validated_signal = np.asarray(
        signal
    )

    if validated_signal.size == 0:
        raise ValueError(
            "Signal cannot be empty."
        )

    if validated_signal.ndim != 1:
        raise ValueError(
            "Signal must be one-dimensional."
        )

    if not np.all(
        np.isfinite(validated_signal)
    ):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    return validated_signal


# =====================================================================
# CONVERT TO COMPLEX SIGNAL
# =====================================================================

def _prepare_complex_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Convert a real or complex signal into a complex signal.

    Complex signals:
        Used directly.

    Real signals:
        Converted into an analytic signal using the Hilbert
        transform.
    """

    validated_signal = _validate_signal(
        signal
    )

    if np.iscomplexobj(
        validated_signal
    ):

        complex_signal = np.asarray(
            validated_signal,
            dtype=np.complex128,
        )

        return complex_signal

    real_signal = np.asarray(
        validated_signal,
        dtype=np.float64,
    )

    analytic_signal = hilbert(
        real_signal
    )

    complex_signal = np.asarray(
        analytic_signal,
        dtype=np.complex128,
    )

    return complex_signal


# =====================================================================
# NORMALIZATION
# =====================================================================

def _normalize_complex_signal(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Normalize a complex signal using root-mean-square magnitude.
    """

    complex_signal = np.asarray(
        signal,
        dtype=np.complex128,
    )

    power = np.mean(
        np.abs(complex_signal) ** 2
    )

    if power <= 0:
        raise ValueError(
            "Cannot normalize a zero-energy signal."
        )

    root_mean_square = np.sqrt(
        power
    )

    normalized_signal = (
        complex_signal
        / root_mean_square
    )

    return np.asarray(
        normalized_signal,
        dtype=np.complex128,
    )


# =====================================================================
# CONSTELLATION EXTRACTION
# =====================================================================

def extract_constellation_points(
    signal: np.ndarray,
    remove_dc: bool = True,
    normalize: bool = True,
    max_points: Optional[int] = 10000,
) -> np.ndarray:
    """
    Extract complex constellation points.

    Parameters
    ----------
    signal:
        Real or complex signal.

    remove_dc:
        Remove the average complex offset.

    normalize:
        Normalize the signal amplitude.

    max_points:
        Maximum number of points returned.

    Returns
    -------
    numpy.ndarray
        Complex constellation points.
    """

    complex_signal = _prepare_complex_signal(
        signal
    )

    if remove_dc:

        dc_value = np.mean(
            complex_signal
        )

        complex_signal = (
            complex_signal
            - dc_value
        )

    if normalize:

        complex_signal = (
            _normalize_complex_signal(
                complex_signal
            )
        )

    if max_points is not None:

        if max_points <= 0:
            raise ValueError(
                "max_points must be greater than zero."
            )

        if complex_signal.size > max_points:

            indices = np.linspace(
                0,
                complex_signal.size - 1,
                max_points,
                dtype=np.int64,
            )

            complex_signal = (
                complex_signal[indices]
            )

    return np.asarray(
        complex_signal,
        dtype=np.complex128,
    )


# =====================================================================
# BASIC STATISTICS
# =====================================================================

def calculate_constellation_statistics(
    constellation_points: np.ndarray,
) -> dict[str, Any]:
    """
    Calculate statistical features from constellation points.
    """

    points = np.asarray(
        constellation_points,
        dtype=np.complex128,
    )

    if points.size == 0:
        raise ValueError(
            "Constellation cannot be empty."
        )

    if points.ndim != 1:
        raise ValueError(
            "Constellation must be one-dimensional."
        )

    if not np.all(
        np.isfinite(points)
    ):
        raise ValueError(
            "Constellation contains NaN or infinite values."
        )

    in_phase = np.asarray(
        np.real(points),
        dtype=np.float64,
    )

    quadrature = np.asarray(
        np.imag(points),
        dtype=np.float64,
    )

    magnitudes = np.asarray(
        np.abs(points),
        dtype=np.float64,
    )

    phases = np.asarray(
        np.angle(points),
        dtype=np.float64,
    )

    mean_i = float(
        np.mean(in_phase)
    )

    mean_q = float(
        np.mean(quadrature)
    )

    i_standard_deviation = float(
        np.std(in_phase)
    )

    q_standard_deviation = float(
        np.std(quadrature)
    )

    mean_magnitude = float(
        np.mean(magnitudes)
    )

    magnitude_standard_deviation = float(
        np.std(magnitudes)
    )

    mean_power = float(
        np.mean(magnitudes ** 2)
    )

    circular_phase_mean = float(
        np.angle(
            np.mean(
                np.exp(1j * phases)
            )
        )
    )

    if points.size > 1:

        covariance_matrix = np.asarray(
            np.cov(
                in_phase,
                quadrature,
            ),
            dtype=np.float64,
        )

        iq_covariance = float(
            covariance_matrix[0, 1]
        )

    else:

        iq_covariance = 0.0

    return {
        "number_of_points": int(
            points.size
        ),

        "mean_i": mean_i,

        "mean_q": mean_q,

        "i_standard_deviation": (
            i_standard_deviation
        ),

        "q_standard_deviation": (
            q_standard_deviation
        ),

        "mean_magnitude": (
            mean_magnitude
        ),

        "magnitude_standard_deviation": (
            magnitude_standard_deviation
        ),

        "mean_power": mean_power,

        "mean_phase_radians": (
            circular_phase_mean
        ),

        "iq_covariance": (
            iq_covariance
        ),
    }


# =====================================================================
# RADIUS STATISTICS
# =====================================================================

def calculate_radius_statistics(
    constellation_points: np.ndarray,
) -> dict[str, Any]:
    """
    Calculate statistics describing constellation radii.
    """

    points = np.asarray(
        constellation_points,
        dtype=np.complex128,
    )

    if points.size == 0:
        raise ValueError(
            "Constellation cannot be empty."
        )

    magnitudes = np.asarray(
        np.abs(points),
        dtype=np.float64,
    )

    percentile_values = np.percentile(
        magnitudes,
        [10, 25, 50, 75, 90],
    )

    return {
        "minimum_radius": float(
            np.min(magnitudes)
        ),

        "maximum_radius": float(
            np.max(magnitudes)
        ),

        "median_radius": float(
            percentile_values[2]
        ),

        "radius_percentile_10": float(
            percentile_values[0]
        ),

        "radius_percentile_25": float(
            percentile_values[1]
        ),

        "radius_percentile_75": float(
            percentile_values[3]
        ),

        "radius_percentile_90": float(
            percentile_values[4]
        ),
    }


# =====================================================================
# CLUSTERING
# =====================================================================

def _run_clustering(
    constellation_points: np.ndarray,
    number_of_clusters: Optional[int] = None,
) -> dict[str, Any]:
    """
    Run the constellation clustering algorithm.
    """

    from .clustering import (
        cluster_constellation,
    )

    if number_of_clusters is None:

        clustering_result = (
            cluster_constellation(
                constellation_points
            )
        )

    else:

        clustering_result = (
            cluster_constellation(
                constellation_points,
                number_of_clusters=(
                    number_of_clusters
                ),
            )
        )

    return {
        "clustering_available": True,
        "cluster_result": clustering_result,
    }


# =====================================================================
# COMPLETE CONSTELLATION ANALYSIS
# =====================================================================

def analyze_constellation(
    signal: np.ndarray,
    remove_dc: bool = True,
    normalize: bool = True,
    max_points: Optional[int] = 10000,
    number_of_clusters: Optional[int] = None,
) -> dict[str, Any]:
    """
    Perform complete constellation analysis.
    """

    constellation_points = (
        extract_constellation_points(
            signal=signal,
            remove_dc=remove_dc,
            normalize=normalize,
            max_points=max_points,
        )
    )

    statistics = (
        calculate_constellation_statistics(
            constellation_points
        )
    )

    radius_statistics = (
        calculate_radius_statistics(
            constellation_points
        )
    )

    clustering = _run_clustering(
        constellation_points,
        number_of_clusters=(
            number_of_clusters
        ),
    )

    in_phase = np.asarray(
        np.real(constellation_points),
        dtype=np.float64,
    )

    quadrature = np.asarray(
        np.imag(constellation_points),
        dtype=np.float64,
    )

    result: dict[str, Any] = {
        "constellation_points": (
            constellation_points
        ),

        "in_phase": in_phase,

        "quadrature": quadrature,

        "statistics": statistics,

        "radius_statistics": (
            radius_statistics
        ),

        "clustering": clustering,
    }

    return result


# =====================================================================
# WAV FILE LOADING
# =====================================================================

def _load_wav_file(
    file_path: Path,
) -> Tuple[np.ndarray, float]:
    """
    Load a WAV file.

    Stereo WAV files are converted to mono.
    """

    sampling_rate, signal = (
        wavfile.read(
            str(file_path)
        )
    )

    loaded_signal = np.asarray(
        signal
    )

    if loaded_signal.ndim == 2:

        loaded_signal = np.mean(
            loaded_signal.astype(
                np.float64
            ),
            axis=1,
        )

    elif loaded_signal.ndim != 1:

        raise ValueError(
            "WAV file must contain one-dimensional "
            "or stereo audio data."
        )

    if np.issubdtype(
        loaded_signal.dtype,
        np.integer,
    ):

        floating_signal = (
            loaded_signal.astype(
                np.float64
            )
        )

        maximum_amplitude = float(
            np.max(
                np.abs(
                    floating_signal
                )
            )
        )

        if maximum_amplitude > 0:

            floating_signal = (
                floating_signal
                / maximum_amplitude
            )

        loaded_signal = floating_signal

    else:

        loaded_signal = (
            loaded_signal.astype(
                np.float64
            )
        )

    return (
        np.asarray(
            loaded_signal,
            dtype=np.float64,
        ),
        float(sampling_rate),
    )


# =====================================================================
# RAW IQ FILE LOADING
# =====================================================================

def _load_iq_file(
    file_path: Path,
    iq_dtype: Any = np.float32,
    iq_order: str = "IQ",
) -> np.ndarray:
    """
    Load an interleaved raw IQ file.

    Expected format:

        I, Q, I, Q, I, Q, ...

    Supported orders:

        IQ
        QI
    """

    raw_data = np.fromfile(
        str(file_path),
        dtype=iq_dtype,
    )

    if raw_data.size == 0:

        raise ValueError(
            "IQ file is empty."
        )

    if raw_data.size % 2 != 0:

        raise ValueError(
            "IQ file must contain an even number "
            "of values."
        )

    first_component = np.asarray(
        raw_data[0::2],
        dtype=np.float64,
    )

    second_component = np.asarray(
        raw_data[1::2],
        dtype=np.float64,
    )

    order = str(
        iq_order
    ).upper()

    if order == "IQ":

        i_samples = first_component
        q_samples = second_component

    elif order == "QI":

        q_samples = first_component
        i_samples = second_component

    else:

        raise ValueError(
            "iq_order must be either 'IQ' or 'QI'."
        )

    complex_signal = (
        i_samples
        + 1j * q_samples
    )

    return np.asarray(
        complex_signal,
        dtype=np.complex128,
    )


# =====================================================================
# FILE-BASED ANALYSIS
# =====================================================================

def analyze_constellation_from_file(
    file_path: Any,
    sampling_rate: Optional[float] = None,
    iq_dtype: Any = np.float32,
    iq_order: str = "IQ",
    remove_dc: bool = True,
    normalize: bool = True,
    max_points: Optional[int] = 10000,
    number_of_clusters: Optional[int] = None,
) -> dict[str, Any]:
    """
    Analyze constellation data directly from a WAV or IQ file.

    For WAV:
        Sampling rate is read from the WAV header.

    For IQ:
        sampling_rate must be supplied by the caller.
    """

    path = Path(
        file_path
    )

    if not path.exists():

        raise FileNotFoundError(
            "File not found: "
            + str(path)
        )

    extension = (
        path.suffix.lower()
    )

    if extension == ".wav":

        signal, file_sampling_rate = (
            _load_wav_file(
                path
            )
        )

        effective_sampling_rate = (
            file_sampling_rate
        )

    elif extension == ".iq":

        if sampling_rate is None:

            raise ValueError(
                "sampling_rate is required "
                "for raw IQ files."
            )

        signal = _load_iq_file(
            path,
            iq_dtype=iq_dtype,
            iq_order=iq_order,
        )

        effective_sampling_rate = float(
            sampling_rate
        )

    else:

        raise ValueError(
            "Unsupported file format. "
            "Only .wav and .iq files are supported."
        )

    result = analyze_constellation(
        signal=signal,
        remove_dc=remove_dc,
        normalize=normalize,
        max_points=max_points,
        number_of_clusters=(
            number_of_clusters
        ),
    )

    result["file"] = str(
        path
    )

    result["file_type"] = extension

    result["sampling_rate_hz"] = float(
        effective_sampling_rate
    )

    return result