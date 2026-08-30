import unittest

import numpy as np

from dsp.constellation.constellation import analyze_constellation
from dsp.constellation.clustering import cluster_constellation


class TestConstellationAndClustering(unittest.TestCase):
    """Tests for constellation extraction and clustering."""

    def setUp(self):
        """
        Create a clean QPSK constellation for testing.

        Four ideal constellation points:
            +1 + 1j
            -1 + 1j
            -1 - 1j
            +1 - 1j
        """

        self.qpsk_points = np.array(
            [
                1 + 1j,
                -1 + 1j,
                -1 - 1j,
                1 - 1j,
            ],
            dtype=np.complex128,
        )

        # Repeat the points so that the clustering algorithm
        # has enough samples to work with.
        self.qpsk_signal = np.tile(
            self.qpsk_points,
            250,
        )

    # ---------------------------------------------------------
    # BASIC CONSTELLATION TEST
    # ---------------------------------------------------------

    def test_constellation_analysis(self):
        """Test basic constellation extraction."""

        result = analyze_constellation(
            self.qpsk_signal,
            number_of_clusters=4,
        )

        self.assertIsInstance(result, dict)

        self.assertIn(
            "constellation_points",
            result,
        )

        self.assertIn(
            "in_phase",
            result,
        )

        self.assertIn(
            "quadrature",
            result,
        )

        self.assertIn(
            "statistics",
            result,
        )

        self.assertIn(
            "clustering",
            result,
        )

    # ---------------------------------------------------------
    # CONSTELLATION POINT TEST
    # ---------------------------------------------------------

    def test_constellation_points(self):
        """Verify that constellation points are complex."""

        result = analyze_constellation(
            self.qpsk_signal,
            number_of_clusters=4,
        )

        points = np.asarray(
            result["constellation_points"]
        )

        self.assertGreater(
            len(points),
            0,
        )

        self.assertTrue(
            np.iscomplexobj(points)
        )

    # ---------------------------------------------------------
    # I/Q EXTRACTION TEST
    # ---------------------------------------------------------

    def test_iq_extraction(self):
        """Test I/Q coordinates returned by constellation analysis."""

        result = analyze_constellation(
            self.qpsk_signal,
            number_of_clusters=4,
        )

        in_phase = np.asarray(
            result["in_phase"]
        )

        quadrature = np.asarray(
            result["quadrature"]
        )

        self.assertEqual(
            len(in_phase),
            len(quadrature),
        )

        self.assertGreater(
            len(in_phase),
            0,
        )

        self.assertTrue(
            np.all(np.isfinite(in_phase))
        )

        self.assertTrue(
            np.all(np.isfinite(quadrature))
        )

    # ---------------------------------------------------------
    # DIRECT CLUSTERING TEST
    # ---------------------------------------------------------

    def test_clustering(self):
        """Test the clustering module directly."""

        coordinates = np.column_stack(
            (
                self.qpsk_signal.real,
                self.qpsk_signal.imag,
            )
        )

        result = cluster_constellation(
            coordinates,
            number_of_clusters=4,
        )

        self.assertIsInstance(
            result,
            dict,
        )

        self.assertIn(
            "labels",
            result,
        )

        self.assertIn(
            "centers",
            result,
        )

        labels = np.asarray(
            result["labels"]
        )

        centers = np.asarray(
            result["centers"]
        )

        self.assertEqual(
            len(labels),
            len(coordinates),
        )

        self.assertEqual(
            centers.shape[0],
            4,
        )

    # ---------------------------------------------------------
    # CLUSTER COUNT TEST
    # ---------------------------------------------------------

    def test_cluster_count(self):
        """Verify that four QPSK clusters are detected."""

        coordinates = np.column_stack(
            (
                self.qpsk_signal.real,
                self.qpsk_signal.imag,
            )
        )

        result = cluster_constellation(
            coordinates,
            number_of_clusters=4,
        )

        labels = np.asarray(
            result["labels"]
        )

        unique_labels = np.unique(
            labels
        )

        self.assertEqual(
            len(unique_labels),
            4,
        )

    # ---------------------------------------------------------
    # CLUSTER CENTERS TEST
    # ---------------------------------------------------------

    def test_cluster_centers(self):
        """Verify that cluster centers are close to QPSK points."""

        coordinates = np.column_stack(
            (
                self.qpsk_signal.real,
                self.qpsk_signal.imag,
            )
        )

        result = cluster_constellation(
            coordinates,
            number_of_clusters=4,
        )

        centers = np.asarray(
            result["centers"],
            dtype=np.float64,
        )

        expected_centers = np.array(
            [
                [1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [1.0, -1.0],
            ],
            dtype=np.float64,
        )

        self.assertEqual(
            centers.shape,
            (4, 2),
        )

        # Every detected center should be close to
        # one of the expected QPSK locations.
        for center in centers:

            distances = np.linalg.norm(
                expected_centers - center,
                axis=1,
            )

            minimum_distance = np.min(
                distances
            )

            self.assertLess(
                minimum_distance,
                0.2,
            )

    # ---------------------------------------------------------
    # NO NaN / INFINITY TEST
    # ---------------------------------------------------------

    def test_no_invalid_values(self):
        """Ensure constellation output contains finite values."""

        result = analyze_constellation(
            self.qpsk_signal,
            number_of_clusters=4,
        )

        points = np.asarray(
            result["constellation_points"]
        )

        self.assertTrue(
            np.all(np.isfinite(points.real))
        )

        self.assertTrue(
            np.all(np.isfinite(points.imag))
        )

    # ---------------------------------------------------------
    # INVALID INPUT TEST
    # ---------------------------------------------------------

    def test_empty_signal(self):
        """Empty input should raise ValueError."""

        empty_signal = np.array(
            [],
            dtype=np.complex128,
        )

        with self.assertRaises(ValueError):

            analyze_constellation(
                empty_signal,
                number_of_clusters=4,
            )

    # ---------------------------------------------------------
    # TOO FEW CLUSTERS TEST
    # ---------------------------------------------------------

    def test_different_cluster_count(self):
        """Test clustering with two clusters."""

        signal = np.array(
            [
                1 + 1j,
                1 + 1j,
                1 + 1j,
                -1 - 1j,
                -1 - 1j,
                -1 - 1j,
            ],
            dtype=np.complex128,
        )

        coordinates = np.column_stack(
            (
                signal.real,
                signal.imag,
            )
        )

        result = cluster_constellation(
            coordinates,
            number_of_clusters=2,
        )

        labels = np.asarray(
            result["labels"]
        )

        centers = np.asarray(
            result["centers"]
        )

        self.assertEqual(
            len(labels),
            len(signal),
        )

        self.assertEqual(
            centers.shape[0],
            2,
        )


if __name__ == "__main__":
    unittest.main()