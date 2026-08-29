import numpy as np

from signal_processing.input.loader import load_signal
from signal_processing.input.validation import validate_samples

from signal_processing.detection.signal_detector import detect_signal

from signal_processing.preprocessing.dc_removal import remove_dc
from signal_processing.preprocessing.filtering import lowpass_filter
from signal_processing.preprocessing.normalization import normalize_iq
from signal_processing.preprocessing.resampling import resample_signal


# ============================================================
# INPUT FILE
# ============================================================

FILE_PATH = r"data\iq\signal_0001_bpsk.iq"

# Sampling rate of the raw IQ file.
#
# IMPORTANT:
# Replace this value with the actual sampling frequency
# from your dataset.csv if it is different.
IQ_SAMPLE_RATE = 1_000_000


# ============================================================
# 1. LOAD SIGNAL
# ============================================================

print("=" * 60)
print("1. LOADING SIGNAL")
print("=" * 60)

if FILE_PATH.lower().endswith(".iq"):

    samples, sample_rate = load_signal(
        FILE_PATH,
        iq_sample_rate=IQ_SAMPLE_RATE,
    )

else:

    samples, sample_rate = load_signal(
        FILE_PATH
    )


print("File:", FILE_PATH)
print("Number of samples:", len(samples))
print("Sample rate:", sample_rate, "Hz")
print("Data type:", samples.dtype)


# ============================================================
# 2. VALIDATE SIGNAL
# ============================================================

print()
print("=" * 60)
print("2. VALIDATING SIGNAL")
print("=" * 60)

validate_samples(
    samples,
    sample_rate,
)

print("Validation: PASSED")


# ============================================================
# 3. SIGNAL DETECTION
# ============================================================

print()
print("=" * 60)
print("3. SIGNAL DETECTION")
print("=" * 60)

try:

    detected_signal, region = detect_signal(
        samples,
        threshold=0.01,
    )

    print("Detection: PASSED")
    print("Detected region:", region)
    print("Detected samples:", len(detected_signal))

except ValueError:

    print(
        "No signal region detected with threshold 0.01."
    )

    print(
        "Using the complete loaded signal for "
        "the preprocessing test."
    )

    detected_signal = samples


# ============================================================
# 4. DC REMOVAL
# ============================================================

print()
print("=" * 60)
print("4. DC REMOVAL")
print("=" * 60)

dc_removed = remove_dc(
    detected_signal
)

print("DC removal: PASSED")

print(
    "Mean after DC removal:",
    np.mean(dc_removed)
)


# ============================================================
# 5. FILTERING
# ============================================================

print()
print("=" * 60)
print("5. FILTERING")
print("=" * 60)

# Test cutoff frequency.
#
# This is only for verifying that the filtering code
# works with the real dataset signal.
#
# It is NOT the final DSP filtering parameter.
cutoff_frequency = sample_rate * 0.2

filtered = lowpass_filter(
    dc_removed,
    sample_rate=sample_rate,
    cutoff_frequency=cutoff_frequency,
)

print("Filtering: PASSED")
print("Cutoff frequency:", cutoff_frequency, "Hz")


# ============================================================
# 6. NORMALIZATION
# ============================================================

print()
print("=" * 60)
print("6. NORMALIZATION")
print("=" * 60)

normalized = normalize_iq(
    filtered
)

print("Normalization: PASSED")

print(
    "Peak magnitude:",
    np.max(np.abs(normalized))
)


# ============================================================
# 7. RESAMPLING
# ============================================================

print()
print("=" * 60)
print("7. RESAMPLING")
print("=" * 60)

# For this first real-data test, keep the same sample rate.
#
# This verifies that the resampling function correctly
# handles a signal when no rate change is required.
target_sample_rate = sample_rate

resampled, new_sample_rate = resample_signal(
    normalized,
    original_sample_rate=sample_rate,
    target_sample_rate=target_sample_rate,
)

print("Resampling: PASSED")
print("Original sample rate:", sample_rate, "Hz")
print("New sample rate:", new_sample_rate, "Hz")
print("Final number of samples:", len(resampled))


# ============================================================
# 8. FINAL VALIDATION
# ============================================================

print()
print("=" * 60)
print("8. FINAL CHECK")
print("=" * 60)

assert isinstance(resampled, np.ndarray)

assert resampled.size > 0

assert np.isfinite(
    resampled.real
).all()

assert np.isfinite(
    resampled.imag
).all()

assert np.isclose(
    new_sample_rate,
    target_sample_rate,
)


# ============================================================
# SUCCESS
# ============================================================

print()
print("=" * 60)
print("PREPROCESSING PIPELINE PASSED")
print("=" * 60)

print()
print("Pipeline:")
print("IQ file")
print("  ↓")
print("Loader")
print("  ↓")
print("Validation")
print("  ↓")
print("Detection")
print("  ↓")
print("DC removal")
print("  ↓")
print("Filtering")
print("  ↓")
print("Normalization")
print("  ↓")
print("Resampling")
print("  ↓")
print("Final validation")
print()
print("SUCCESS")