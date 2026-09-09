import numpy as np
import pytest

from backend.ml.evaluation.robustness import (
    STANDARD_PERTURBATIONS,
    apply_gaussian_noise,
    apply_illumination_shift,
    apply_jpeg_compression,
    apply_peripheral_occlusion,
    apply_rotation,
    apply_sensor_blur,
    evaluate_perturbation_series,
    run_full_robustness_suite,
)


@pytest.fixture
def synthetic_fingerprint_image():
    # Synthetic 96x96 BGR ridge pattern
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[:, ::4, :] = 255
    img[::4, :, :] = 180
    return img


def test_apply_rotation(synthetic_fingerprint_image):
    rotated = apply_rotation(synthetic_fingerprint_image, angle_degrees=15.0)
    assert rotated.shape == synthetic_fingerprint_image.shape
    assert rotated.dtype == np.uint8
    # Image should have changed from original
    assert not np.array_equal(rotated, synthetic_fingerprint_image)


def test_apply_gaussian_noise(synthetic_fingerprint_image):
    noisy = apply_gaussian_noise(synthetic_fingerprint_image, sigma=25.0, seed=42)
    assert noisy.shape == synthetic_fingerprint_image.shape
    assert noisy.dtype == np.uint8
    assert not np.array_equal(noisy, synthetic_fingerprint_image)

    # Zero noise returns copy
    zero_noise = apply_gaussian_noise(synthetic_fingerprint_image, sigma=0.0)
    assert np.array_equal(zero_noise, synthetic_fingerprint_image)


def test_apply_sensor_blur(synthetic_fingerprint_image):
    blurred = apply_sensor_blur(synthetic_fingerprint_image, kernel_size=5)
    assert blurred.shape == synthetic_fingerprint_image.shape
    assert blurred.dtype == np.uint8
    assert not np.array_equal(blurred, synthetic_fingerprint_image)

    # Kernel size 1 returns copy
    noop_blur = apply_sensor_blur(synthetic_fingerprint_image, kernel_size=1)
    assert np.array_equal(noop_blur, synthetic_fingerprint_image)


def test_apply_illumination_shift(synthetic_fingerprint_image):
    shifted = apply_illumination_shift(synthetic_fingerprint_image, alpha=1.2, beta=10.0)
    assert shifted.shape == synthetic_fingerprint_image.shape
    assert shifted.dtype == np.uint8
    assert not np.array_equal(shifted, synthetic_fingerprint_image)


def test_apply_peripheral_occlusion(synthetic_fingerprint_image):
    occluded = apply_peripheral_occlusion(synthetic_fingerprint_image, occlusion_ratio=0.25, fill_value=255)
    assert occluded.shape == synthetic_fingerprint_image.shape
    assert occluded.dtype == np.uint8

    # Border should be masked with fill_value
    assert np.all(occluded[:10, :, :] == 255)
    assert np.all(occluded[:, :10, :] == 255)


def test_apply_jpeg_compression(synthetic_fingerprint_image):
    compressed = apply_jpeg_compression(synthetic_fingerprint_image, quality=25)
    assert compressed.shape == synthetic_fingerprint_image.shape
    assert compressed.dtype == np.uint8


def test_evaluate_perturbation_series(synthetic_fingerprint_image):
    def mock_predict_fn(img: np.ndarray) -> dict:
        # Mock predictor that drops confidence when noisy or blurred
        variance = float(np.var(img))
        confidence = max(0.40, min(0.95, variance / 5000.0))
        status = "ACCEPTED" if confidence >= 0.60 else "PREDICTION_WITHHELD"
        return {
            "predicted_class": "A+",
            "confidence": confidence * 100.0,
            "decision_status": status,
        }

    samples = [(synthetic_fingerprint_image, "A+"), (synthetic_fingerprint_image, "B+")]

    results = evaluate_perturbation_series(
        predict_fn=mock_predict_fn,
        test_samples=samples,
        perturbation_category="gaussian_noise",
    )

    assert results["category"] == "gaussian_noise"
    assert results["sample_count"] == 2
    assert len(results["steps"]) == len(STANDARD_PERTURBATIONS["gaussian_noise"]) + 1  # includes baseline
    assert results["steps"][0]["perturbation"] == "baseline"
    assert "accuracy" in results["steps"][0]
    assert "mean_confidence" in results["steps"][0]
    assert "abstention_rate" in results["steps"][0]


def test_run_full_robustness_suite(synthetic_fingerprint_image):
    def mock_predict_fn(img: np.ndarray) -> dict:
        return {
            "predicted_class": "A+",
            "confidence": 85.0,
            "decision_status": "ACCEPTED",
        }

    samples = [(synthetic_fingerprint_image, "A+")]
    suite_report = run_full_robustness_suite(mock_predict_fn, samples)

    for cat in ["rotation", "gaussian_noise", "sensor_blur", "illumination", "partial_occlusion", "jpeg_compression"]:
        assert cat in suite_report
        assert len(suite_report[cat]["steps"]) > 0
