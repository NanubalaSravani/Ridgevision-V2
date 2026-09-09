from collections.abc import Callable
from typing import Any, Optional

import cv2
import numpy as np


def apply_rotation(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Applies in-plane angular rotation around center with border replication."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return rotated


def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Adds zero-mean Gaussian sensor noise with specified standard deviation sigma."""
    if sigma <= 0:
        return image.copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_sensor_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Simulates optical / sensor defocus blur using a Gaussian filter."""
    if kernel_size <= 1:
        return image.copy()
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_illumination_shift(
    image: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> np.ndarray:
    """Applies contrast scaling (alpha) and brightness offset (beta).

    output = clip(alpha * image + beta, 0, 255)
    """
    shifted = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    return shifted


def apply_peripheral_occlusion(
    image: np.ndarray,
    occlusion_ratio: float = 0.20,
    fill_value: int = 255,
) -> np.ndarray:
    """Simulates partial capture / peripheral scanner boundary occlusion.

    occlusion_ratio: proportion of outer margin to mask (e.g., 0.20 masks outer 20% on each side).
    fill_value: background fill (default 255 for scanner white, or 0 for black).
    """
    occlusion_ratio = float(np.clip(occlusion_ratio, 0.0, 0.49))
    if occlusion_ratio <= 0.0:
        return image.copy()

    h, w = image.shape[:2]
    pad_h = int(h * occlusion_ratio)
    pad_w = int(w * occlusion_ratio)

    occluded = np.full_like(image, fill_value)
    occluded[pad_h : h - pad_h, pad_w : w - pad_w] = image[pad_h : h - pad_h, pad_w : w - pad_w]
    return occluded


def apply_jpeg_compression(image: np.ndarray, quality: int = 80) -> np.ndarray:
    """Simulates lossy JPEG compression artifacts at specified quality factor (1-100)."""
    quality = int(np.clip(quality, 1, 100))
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, enc = cv2.imencode(".jpg", image, encode_params)
    if not success:
        return image.copy()
    decoded = cv2.imdecode(enc, cv2.IMREAD_UNCHANGED)
    return decoded


STANDARD_PERTURBATIONS: dict[str, list[dict[str, Any]]] = {
    "rotation": [
        {"name": "rot_-30", "fn": lambda img: apply_rotation(img, -30.0), "param": -30},
        {"name": "rot_-15", "fn": lambda img: apply_rotation(img, -15.0), "param": -15},
        {"name": "rot_-5", "fn": lambda img: apply_rotation(img, -5.0), "param": -5},
        {"name": "rot_+5", "fn": lambda img: apply_rotation(img, 5.0), "param": 5},
        {"name": "rot_+15", "fn": lambda img: apply_rotation(img, 15.0), "param": 15},
        {"name": "rot_+30", "fn": lambda img: apply_rotation(img, 30.0), "param": 30},
    ],
    "gaussian_noise": [
        {"name": "noise_sigma_10", "fn": lambda img: apply_gaussian_noise(img, 10.0), "param": 10},
        {"name": "noise_sigma_25", "fn": lambda img: apply_gaussian_noise(img, 25.0), "param": 25},
        {"name": "noise_sigma_50", "fn": lambda img: apply_gaussian_noise(img, 50.0), "param": 50},
    ],
    "sensor_blur": [
        {"name": "blur_k3", "fn": lambda img: apply_sensor_blur(img, 3), "param": 3},
        {"name": "blur_k5", "fn": lambda img: apply_sensor_blur(img, 5), "param": 5},
        {"name": "blur_k7", "fn": lambda img: apply_sensor_blur(img, 7), "param": 7},
    ],
    "illumination": [
        {"name": "contrast_low_0.7", "fn": lambda img: apply_illumination_shift(img, alpha=0.7, beta=0), "param": 0.7},
        {"name": "contrast_high_1.3", "fn": lambda img: apply_illumination_shift(img, alpha=1.3, beta=0), "param": 1.3},
        {"name": "bright_minus_30", "fn": lambda img: apply_illumination_shift(img, alpha=1.0, beta=-30), "param": -30},
        {"name": "bright_plus_30", "fn": lambda img: apply_illumination_shift(img, alpha=1.0, beta=30), "param": 30},
    ],
    "partial_occlusion": [
        {"name": "occlusion_20pct", "fn": lambda img: apply_peripheral_occlusion(img, 0.20), "param": 0.20},
        {"name": "occlusion_30pct", "fn": lambda img: apply_peripheral_occlusion(img, 0.30), "param": 0.30},
        {"name": "occlusion_40pct", "fn": lambda img: apply_peripheral_occlusion(img, 0.40), "param": 0.40},
    ],
    "jpeg_compression": [
        {"name": "jpeg_q80", "fn": lambda img: apply_jpeg_compression(img, 80), "param": 80},
        {"name": "jpeg_q50", "fn": lambda img: apply_jpeg_compression(img, 50), "param": 50},
        {"name": "jpeg_q25", "fn": lambda img: apply_jpeg_compression(img, 25), "param": 25},
    ],
}


def evaluate_perturbation_series(
    predict_fn: Callable[[np.ndarray], dict[str, Any]],
    test_samples: list[tuple[np.ndarray, str]],
    perturbation_category: str = "gaussian_noise",
    perturbations: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Evaluates prediction accuracy, confidence, and abstention rate across a series of perturbations.

    predict_fn: takes BGR image array and returns dict with keys:
                'predicted_class', 'confidence' (or 'top_confidence'),
                and optionally 'decision_status' ("ACCEPTED" or "PREDICTION_WITHHELD").
    test_samples: list of (image_bgr, true_label)
    """
    if perturbations is None:
        if perturbation_category not in STANDARD_PERTURBATIONS:
            raise ValueError(f"Unknown perturbation category: {perturbation_category}. Available: {list(STANDARD_PERTURBATIONS.keys())}")
        perturbations = STANDARD_PERTURBATIONS[perturbation_category]

    n_samples = len(test_samples)
    if n_samples == 0:
        return {"category": perturbation_category, "steps": []}

    # Evaluate baseline (unperturbed)
    base_correct = 0
    base_confs = []
    base_abstain = 0
    for img, label in test_samples:
        pred = predict_fn(img)
        pred_class = pred.get("predicted_class")
        conf = float(pred.get("confidence", pred.get("top_confidence", 0.0)))
        if conf > 1.0:
            conf /= 100.0  # normalize if in percentage
        base_confs.append(conf)
        if pred_class == label:
            base_correct += 1
        if pred.get("decision_status") == "PREDICTION_WITHHELD":
            base_abstain += 1

    steps_results = [
        {
            "perturbation": "baseline",
            "parameter": 0,
            "accuracy": round(base_correct / n_samples, 4),
            "mean_confidence": round(float(np.mean(base_confs)), 4),
            "abstention_rate": round(base_abstain / n_samples, 4),
        }
    ]

    for p in perturbations:
        p_name = p["name"]
        p_fn = p["fn"]
        param_val = p.get("param", 0)

        correct = 0
        confs = []
        abstentions = 0

        for img, label in test_samples:
            perturbed_img = p_fn(img)
            pred = predict_fn(perturbed_img)
            pred_class = pred.get("predicted_class")
            conf = float(pred.get("confidence", pred.get("top_confidence", 0.0)))
            if conf > 1.0:
                conf /= 100.0
            confs.append(conf)
            if pred_class == label:
                correct += 1
            if pred.get("decision_status") == "PREDICTION_WITHHELD":
                abstentions += 1

        steps_results.append(
            {
                "perturbation": p_name,
                "parameter": param_val,
                "accuracy": round(correct / n_samples, 4),
                "mean_confidence": round(float(np.mean(confs)), 4),
                "abstention_rate": round(abstentions / n_samples, 4),
            }
        )

    return {
        "category": perturbation_category,
        "sample_count": n_samples,
        "steps": steps_results,
    }


def run_full_robustness_suite(
    predict_fn: Callable[[np.ndarray], dict[str, Any]],
    test_samples: list[tuple[np.ndarray, str]],
) -> dict[str, Any]:
    """Runs all 6 standard perturbation categories and compiles a comprehensive robustness report."""
    results = {}
    for cat in STANDARD_PERTURBATIONS:
        results[cat] = evaluate_perturbation_series(predict_fn, test_samples, perturbation_category=cat)
    return results
