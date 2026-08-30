"""
Constellation clustering utilities.

This module provides lightweight clustering for communication-signal
constellation points.

Supported input:
    - Complex NumPy arrays
    - Nx2 real-valued coordinate arrays

The implementation uses NumPy only.
No scikit-learn dependency is required.

Main functions:
    cluster_constellation()
    select_best_gmm()
    estimate_number_of_clusters()

The name "select_best_gmm" is retained for compatibility with earlier
versions of the project, although the implementation uses a lightweight
K-Means-style clustering algorithm rather than sklearn GaussianMixture.
"""

from typing import Any, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------

def _prepare_coordinates(
    constellation_points: np.ndarray,
) -> np.ndarray:
    """
    Convert constellation points into an Nx2 floating-point array.

    Accepted inputs:

        Complex:
            [1+1j, 1-1j, -1+1j, -1-1j]

        Coordinates:
            [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    """

    points = np.asarray(constellation_points)

    if points.size == 0:
        raise ValueError("Constellation points cannot be empty.")

    if not np.all(np.isfinite(points)):
        raise ValueError(
            "Constellation points contain NaN or infinite values."
        )

    if np.iscomplexobj(points):

        points = points.reshape(-1)

        coordinates = np.column_stack(
            (
                np.real(points),
                np.imag(points),
            )
        )

    else:

        points = np.asarray(points, dtype=np.float64)

        if points.ndim == 1:

            if points.size % 2 != 0:
                raise ValueError(
                    "One-dimensional real constellation data "
                    "must contain an even number of values."
                )

            coordinates = points.reshape(-1, 2)

        elif points.ndim == 2 and points.shape[1] == 2:

            coordinates = points

        else:

            raise ValueError(
                "Real constellation points must have shape (N, 2)."
            )

    coordinates = np.asarray(
        coordinates,
        dtype=np.float64,
    )

    if coordinates.shape[0] == 0:
        raise ValueError("No constellation points were provided.")

    return coordinates


# ---------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------

def _squared_distance_matrix(
    points: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """
    Calculate squared Euclidean distances.

    Returns:
        Matrix with shape:

            (number_of_points, number_of_centers)
    """

    difference = (
        points[:, np.newaxis, :]
        - centers[np.newaxis, :, :]
    )

    return np.sum(
        difference * difference,
        axis=2,
    )


# ---------------------------------------------------------------------
# Initial cluster centers
# ---------------------------------------------------------------------

def _initialize_centers(
    coordinates: np.ndarray,
    number_of_clusters: int,
    random_state: int,
) -> np.ndarray:
    """
    Initialize cluster centers using a deterministic k-means++ style
    procedure.
    """

    number_of_points = coordinates.shape[0]

    if number_of_clusters > number_of_points:
        number_of_clusters = number_of_points

    rng = np.random.default_rng(random_state)

    centers = np.empty(
        (number_of_clusters, 2),
        dtype=np.float64,
    )

    first_index = int(
        rng.integers(0, number_of_points)
    )

    centers[0] = coordinates[first_index]

    if number_of_clusters == 1:
        return centers

    minimum_distances = np.sum(
        (
            coordinates
            - centers[0]
        )
        ** 2,
        axis=1,
    )

    for center_index in range(1, number_of_clusters):

        total_distance = float(
            np.sum(minimum_distances)
        )

        if total_distance <= 0.0:

            candidate_index = int(
                rng.integers(0, number_of_points)
            )

        else:

            probabilities = (
                minimum_distances
                / total_distance
            )

            candidate_index = int(
                rng.choice(
                    number_of_points,
                    p=probabilities,
                )
            )

        centers[center_index] = (
            coordinates[candidate_index]
        )

        new_distances = np.sum(
            (
                coordinates
                - centers[center_index]
            )
            ** 2,
            axis=1,
        )

        minimum_distances = np.minimum(
            minimum_distances,
            new_distances,
        )

    return centers


# ---------------------------------------------------------------------
# K-Means implementation
# ---------------------------------------------------------------------

def _fit_kmeans(
    coordinates: np.ndarray,
    number_of_clusters: int,
    random_state: int = 42,
    max_iterations: int = 100,
) -> Tuple[np.ndarray, np.ndarray, float]:

    """
    Fit a lightweight K-Means clustering model.

    Returns:
        centers
        labels
        inertia
    """

    number_of_points = coordinates.shape[0]

    number_of_clusters = int(
        max(
            1,
            min(
                number_of_clusters,
                number_of_points,
            ),
        )
    )

    centers = _initialize_centers(
        coordinates,
        number_of_clusters,
        random_state,
    )

    labels = np.zeros(
        number_of_points,
        dtype=np.int64,
    )

    for _ in range(max_iterations):

        distances = _squared_distance_matrix(
            coordinates,
            centers,
        )

        new_labels = np.argmin(
            distances,
            axis=1,
        ).astype(np.int64)

        new_centers = np.zeros_like(
            centers
        )

        for cluster_index in range(
            number_of_clusters
        ):

            cluster_points = coordinates[
                new_labels == cluster_index
            ]

            if cluster_points.shape[0] == 0:

                # Reinitialize an empty cluster using
                # the point farthest from its current center.
                point_distances = np.min(
                    distances,
                    axis=1,
                )

                farthest_index = int(
                    np.argmax(point_distances)
                )

                new_centers[
                    cluster_index
                ] = coordinates[farthest_index]

            else:

                new_centers[
                    cluster_index
                ] = np.mean(
                    cluster_points,
                    axis=0,
                )

        if np.array_equal(
            labels,
            new_labels,
        ):
            centers = new_centers
            labels = new_labels
            break

        center_shift = float(
            np.max(
                np.linalg.norm(
                    new_centers - centers,
                    axis=1,
                )
            )
        )

        centers = new_centers
        labels = new_labels

        if center_shift < 1e-8:
            break

    final_distances = _squared_distance_matrix(
        coordinates,
        centers,
    )

    point_distances = final_distances[
        np.arange(number_of_points),
        labels,
    ]

    inertia = float(
        np.sum(point_distances)
    )

    return (
        centers,
        labels,
        inertia,
    )


# ---------------------------------------------------------------------
# Automatic cluster estimation
# ---------------------------------------------------------------------

def estimate_number_of_clusters(
    coordinates: np.ndarray,
    minimum_clusters: int = 1,
    maximum_clusters: Optional[int] = None,
) -> int:
    """
    Estimate a sensible number of constellation clusters.

    The estimate is based on the number of distinct constellation
    locations after quantizing very small numerical differences.

    Examples:

        BPSK  -> approximately 2
        QPSK  -> approximately 4
        16QAM -> approximately 16

    This function is intentionally conservative so that noisy
    constellation points do not create hundreds of clusters.
    """

    points = _prepare_coordinates(
        coordinates
    )

    number_of_points = points.shape[0]

    minimum_clusters = int(
        max(1, minimum_clusters)
    )

    if maximum_clusters is None:
        maximum_clusters = min(
            64,
            number_of_points,
        )
    else:
        maximum_clusters = int(
            max(
                minimum_clusters,
                min(
                    maximum_clusters,
                    number_of_points,
                ),
            )
        )

    if number_of_points == 1:
        return 1

    # Estimate noise scale using nearest-neighbour distances.
    distance_matrix = _squared_distance_matrix(
        points,
        points,
    )

    np.fill_diagonal(
        distance_matrix,
        np.inf,
    )

    nearest_distances = np.sqrt(
        np.min(
            distance_matrix,
            axis=1,
        )
    )

    median_nearest_distance = float(
        np.median(nearest_distances)
    )

    if not np.isfinite(
        median_nearest_distance
    ) or median_nearest_distance <= 0.0:

        return int(
            min(
                maximum_clusters,
                max(
                    minimum_clusters,
                    1,
                ),
            )
        )

    # Try candidate cluster counts and use an elbow-style criterion.
    candidate_max = min(
        maximum_clusters,
        max(
            minimum_clusters,
            16,
        ),
    )

    candidate_scores = []

    previous_inertia: Optional[float] = None

    for cluster_count in range(
        minimum_clusters,
        candidate_max + 1,
    ):

        _, _, inertia = _fit_kmeans(
            points,
            cluster_count,
            random_state=42,
            max_iterations=50,
        )

        if previous_inertia is None:

            improvement = 0.0

        elif previous_inertia > 0.0:

            improvement = (
                previous_inertia - inertia
            ) / previous_inertia

        else:

            improvement = 0.0

        candidate_scores.append(
            (
                cluster_count,
                inertia,
                improvement,
            )
        )

        previous_inertia = inertia

    if not candidate_scores:
        return minimum_clusters

    # Select the first cluster count where the additional cluster
    # provides only a small improvement.
    selected_clusters = int(
        candidate_scores[-1][0]
    )

    for (
        cluster_count,
        _inertia,
        improvement,
    ) in candidate_scores[1:]:

        if improvement < 0.10:

            selected_clusters = int(
                cluster_count - 1
            )

            break

    selected_clusters = int(
        max(
            minimum_clusters,
            min(
                selected_clusters,
                maximum_clusters,
            ),
        )
    )

    return selected_clusters


# ---------------------------------------------------------------------
# Cluster statistics
# ---------------------------------------------------------------------

def _calculate_cluster_sizes(
    labels: np.ndarray,
    number_of_clusters: int,
) -> np.ndarray:
    """
    Calculate the number of points in every cluster.
    """

    number_of_clusters = int(
        max(1, number_of_clusters)
    )

    sizes = np.bincount(
        labels.astype(np.int64),
        minlength=number_of_clusters,
    )

    return np.asarray(
        sizes,
        dtype=np.int64,
    )


def _calculate_cluster_radii(
    coordinates: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """
    Calculate RMS radius for every cluster.
    """

    number_of_clusters = int(
        centers.shape[0]
    )

    radii = np.zeros(
        number_of_clusters,
        dtype=np.float64,
    )

    for cluster_index in range(
        number_of_clusters
    ):

        cluster_points = coordinates[
            labels == cluster_index
        ]

        if cluster_points.shape[0] == 0:
            radii[cluster_index] = 0.0
            continue

        distances = np.linalg.norm(
            cluster_points
            - centers[cluster_index],
            axis=1,
        )

        radii[cluster_index] = float(
            np.sqrt(
                np.mean(
                    distances ** 2
                )
            )
        )

    return radii


def _calculate_cluster_quality(
    coordinates: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Calculate a simple normalized clustering quality score.

    Higher is better.
    """

    number_of_clusters = int(
        centers.shape[0]
    )

    if number_of_clusters <= 1:
        return 1.0

    radii = _calculate_cluster_radii(
        coordinates,
        centers,
        labels,
    )

    within_cluster_spread = float(
        np.mean(radii)
    )

    if within_cluster_spread <= 0.0:
        return 1.0

    center_distances = _squared_distance_matrix(
        centers,
        centers,
    )

    np.fill_diagonal(
        center_distances,
        np.inf,
    )

    minimum_center_distance = float(
        np.sqrt(
            np.min(center_distances)
        )
    )

    if minimum_center_distance <= 0.0:
        return 0.0

    ratio = (
        minimum_center_distance
        / within_cluster_spread
    )

    # Convert to approximately 0-1.
    quality = ratio / (
        ratio + 1.0
    )

    return float(
        np.clip(
            quality,
            0.0,
            1.0,
        )
    )


# ---------------------------------------------------------------------
# Main clustering function
# ---------------------------------------------------------------------

def cluster_constellation(
    constellation_points: np.ndarray,
    number_of_clusters: Optional[int] = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Cluster constellation points.

    Parameters
    ----------
    constellation_points:
        Complex constellation samples or Nx2 coordinates.

    number_of_clusters:
        Desired number of clusters.

        If None, the number is estimated automatically.

    random_state:
        Seed for deterministic clustering.

    Returns
    -------
    dict
        Contains:

            cluster_centers
            labels
            number_of_clusters
            cluster_sizes
            cluster_radii
            inertia
            quality_score
    """

    coordinates = _prepare_coordinates(
        constellation_points
    )

    number_of_points = int(
        coordinates.shape[0]
    )

    if number_of_clusters is None:

        estimated_clusters = (
            estimate_number_of_clusters(
                coordinates,
                minimum_clusters=1,
                maximum_clusters=min(
                    64,
                    number_of_points,
                ),
            )
        )

        number_of_clusters = int(
            estimated_clusters
        )

    else:

        number_of_clusters = int(
            number_of_clusters
        )

        if number_of_clusters < 1:
            raise ValueError(
                "number_of_clusters must be at least 1."
            )

        if number_of_clusters > number_of_points:
            number_of_clusters = number_of_points

    (
        centers,
        labels,
        inertia,
    ) = _fit_kmeans(
        coordinates,
        number_of_clusters,
        random_state=int(random_state),
    )

    cluster_sizes = _calculate_cluster_sizes(
        labels,
        int(number_of_clusters),
    )

    cluster_radii = _calculate_cluster_radii(
        coordinates,
        centers,
        labels,
    )

    quality_score = _calculate_cluster_quality(
        coordinates,
        centers,
        labels,
    )

    complex_centers = (
        centers[:, 0]
        + 1j * centers[:, 1]
    )
    result: dict[str, Any] = {
    # Public/test API: Nx2 real-valued constellation centers
    "centers": np.asarray(
        centers,
        dtype=np.float64,
    ),

    # Backward-compatible complex representation
    "cluster_centers": np.asarray(
        complex_centers,
        dtype=np.complex128,
    ),

    # Explicit Nx2 representation
    "cluster_centers_xy": np.asarray(
        centers,
        dtype=np.float64,
    ),

    "labels": np.asarray(
        labels,
        dtype=np.int64,
    ),

    "number_of_clusters": int(
        number_of_clusters
    ),

    "cluster_sizes": np.asarray(
        cluster_sizes,
        dtype=np.int64,
    ),

    "cluster_radii": np.asarray(
        cluster_radii,
        dtype=np.float64,
    ),

    "inertia": float(
        inertia
    ),

    "quality_score": float(
        quality_score
    ),
}

    return result


# ---------------------------------------------------------------------
# Compatibility function
# ---------------------------------------------------------------------

def select_best_gmm(
    coordinates: np.ndarray,
    minimum_clusters: int = 1,
    maximum_clusters: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[Any, float]:
    """
    Compatibility wrapper for the previous GMM-based implementation.

    The project previously used a GaussianMixture model. To avoid a
    scikit-learn dependency, this function now selects the best
    K-Means-style clustering result.

    Returns
    -------
    model:
        Dictionary containing cluster information.

    score:
        Clustering quality score.
    """

    points = _prepare_coordinates(
        coordinates
    )

    number_of_points = int(
        points.shape[0]
    )

    minimum_clusters = int(
        max(
            1,
            minimum_clusters,
        )
    )

    if maximum_clusters is None:

        maximum_clusters = min(
            16,
            number_of_points,
        )

    else:

        maximum_clusters = int(
            max(
                minimum_clusters,
                min(
                    maximum_clusters,
                    number_of_points,
                ),
            )
        )

    best_result: Optional[dict[str, Any]] = None
    best_score = -np.inf

    for cluster_count in range(
        minimum_clusters,
        maximum_clusters + 1,
    ):

        current_result = cluster_constellation(
            points,
            number_of_clusters=cluster_count,
            random_state=int(random_state),
        )

        current_score = float(
            current_result["quality_score"]
        )

        if (
            best_result is None
            or current_score > best_score
        ):

            best_result = current_result
            best_score = current_score

    if best_result is None:

        best_result = cluster_constellation(
            points,
            number_of_clusters=minimum_clusters,
            random_state=int(random_state),
        )

        best_score = float(
            best_result["quality_score"]
        )

    return (
        best_result,
        float(best_score),
    )


# ---------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------

def get_cluster_centers(
    constellation_points: np.ndarray,
    number_of_clusters: Optional[int] = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Return only the complex cluster centers.
    """

    result = cluster_constellation(
        constellation_points,
        number_of_clusters=number_of_clusters,
        random_state=random_state,
    )

    centers = np.asarray(
        result["cluster_centers"],
        dtype=np.complex128,
    )

    return centers