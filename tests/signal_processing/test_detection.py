import numpy as np

from signal_processing.detection.energy_detection import (
    calculate_signal_power,
    calculate_average_power,
    calculate_energy,
    detect_energy,
)

from signal_processing.detection.segmentation import (
    find_signal_region,
    extract_signal,
)

from signal_processing.detection.signal_detector import (
    detect_signal,
)


def test_energy_calculation():
    samples = np.array(
        [1 + 0j, 2 + 0j, 3 + 0j],
        dtype=np.complex64,
    )

    power = calculate_signal_power(samples)

    assert np.allclose(
        power,
        [1, 4, 9],
    )

    # Floating-point calculations should be compared
    # using tolerance rather than exact equality.
    assert np.isclose(
        calculate_average_power(samples),
        14 / 3,
    )

    assert np.isclose(
        calculate_energy(samples),
        14,
    )


def test_energy_detection():
    samples = np.array(
        [0 + 0j, 1 + 0j, 3 + 0j, 0 + 0j],
        dtype=np.complex64,
    )

    mask = detect_energy(
        samples,
        threshold=2,
    )

    expected = np.array(
        [False, False, True, False]
    )

    assert np.array_equal(
        mask,
        expected,
    )


def test_find_signal_region():
    mask = np.array(
        [False, False, True, True, True, False]
    )

    region = find_signal_region(mask)

    assert region == (2, 5)


def test_extract_signal():
    samples = np.array(
        [0, 0, 1, 2, 3, 0],
        dtype=np.complex64,
    )

    mask = np.array(
        [False, False, True, True, True, False]
    )

    signal = extract_signal(
        samples,
        mask,
    )

    assert np.array_equal(
        signal,
        samples[2:5],
    )


def test_complete_signal_detector():
    samples = np.concatenate(
        [
            np.zeros(
                10,
                dtype=np.complex64,
            ),
            np.ones(
                50,
                dtype=np.complex64,
            ),
            np.zeros(
                10,
                dtype=np.complex64,
            ),
        ]
    )

    detected_signal, region = detect_signal(
        samples,
        threshold=0.5,
    )

    assert region == (10, 60)

    assert len(detected_signal) == 50

    assert np.allclose(
        detected_signal,
        np.ones(
            50,
            dtype=np.complex64,
        ),
    )