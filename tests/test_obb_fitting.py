import cv2
import numpy as np
import pytest

from main.obb_fitting import fit_obb, fit_obb_points


def _oblique_contour():
    theta = np.deg2rad(32.0)
    rotation = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    body = np.asarray(
        [[-12, -3], [-12, 3], [-4, -3], [-4, 3], [4, -3], [4, 3], [12, -3], [12, 3]],
        dtype=np.float32,
    )
    protrusions = np.asarray([[16, -7], [17, -6], [15, 7]], dtype=np.float32)
    return np.concatenate([body, protrusions], axis=0) @ rotation.T


def test_min_area_mode_keeps_opencv_rectangle():
    points = _oblique_contour()
    expected = cv2.minAreaRect(points)
    rect, _ = fit_obb(points, "min_area")

    assert rect == expected
    np.testing.assert_array_equal(fit_obb_points(points, "min_area"), cv2.boxPoints(expected))


def test_pca_mode_aligns_to_first_component_and_contains_contour():
    points = _oblique_contour()
    rect, axis = fit_obb(points, "pca")
    covariance = np.cov(points.astype(np.float64).T)
    _, eigenvectors = np.linalg.eigh(covariance)
    expected_axis = eigenvectors[:, -1]
    assert abs(float(np.dot(axis, expected_axis))) > 0.999

    center = np.asarray(rect[0], dtype=np.float64)
    perpendicular = np.asarray([-axis[1], axis[0]], dtype=np.float64)
    relative = points.astype(np.float64) - center
    along = relative @ axis
    across = relative @ perpendicular
    assert np.max(np.abs(along)) <= float(rect[1][0]) / 2.0 + 1e-5
    assert np.max(np.abs(across)) <= float(rect[1][1]) / 2.0 + 1e-5

    fitted_points = fit_obb_points(points, "pca")
    assert fitted_points.shape == (4, 2)
    assert np.isfinite(fitted_points).all()


def test_unknown_fit_mode_is_rejected():
    with pytest.raises(ValueError, match="OBB_FIT_MODE"):
        fit_obb(_oblique_contour(), "ellipse")
