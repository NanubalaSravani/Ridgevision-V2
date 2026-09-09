import cv2
import numpy as np

from backend.ml.explainability.grad_cam import encode_bgr_to_data_url
from backend.ml.explainability.orientation import compute_orientation_field, singularity_map


def _render_oaas_visualization(
    enhanced_gray: np.ndarray,
    singularity: np.ndarray,
    high_attention_mask: np.ndarray,
) -> str:
    """Renders the Tier 2 image: the singularity/curvature map with white outline

    marking regions that the gradient saliency map considered most important.
    """
    h, w = enhanced_gray.shape[:2]

    singularity_full = cv2.resize(
        np.clip(singularity, 0.0, 1.0).astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_CUBIC,
    )
    singularity_u8 = (singularity_full * 255).astype(np.uint8)
    singularity_color = cv2.applyColorMap(singularity_u8, cv2.COLORMAP_VIRIDIS)

    base = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.5, singularity_color, 0.5, 0)

    mask_full = cv2.resize(
        (high_attention_mask.astype(np.uint8) * 255),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return encode_bgr_to_data_url(overlay)


def orientation_attention_alignment(
    enhanced_gray: np.ndarray,
    attention_map_gray: np.ndarray,
    block_size: int = 16,
) -> dict:
    """Orientation-Attention Alignment Score (OAAS).

    Correlates neural attention/saliency intensity with dermatoglyphic singularity
    likelihood derived from the continuous orientation field.
    """
    fields = compute_orientation_field(enhanced_gray, block_size=block_size)
    singularity = singularity_map(fields["coherence"])

    attention_resized = cv2.resize(
        attention_map_gray.astype(np.float32),
        (singularity.shape[1], singularity.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    a_lo, a_hi = float(attention_resized.min()), float(attention_resized.max())
    attention_norm = (attention_resized - a_lo) / (a_hi - a_lo + 1e-6)

    flat_a = attention_norm.flatten()
    flat_s = singularity.flatten()

    if flat_a.std() < 1e-6 or flat_s.std() < 1e-6:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(flat_a, flat_s)[0, 1])

    if attention_norm.size:
        threshold = np.percentile(attention_norm, 75)
        high_attention_mask = attention_norm > threshold
        overlap_ratio = float(
            singularity[high_attention_mask].mean() if high_attention_mask.any() else 0.0
        )
    else:
        high_attention_mask = np.zeros_like(attention_norm, dtype=bool)
        overlap_ratio = 0.0

    visualization_b64 = _render_oaas_visualization(enhanced_gray, singularity, high_attention_mask)

    return {
        "alignment_correlation": round(correlation, 3),
        "high_attention_singularity_overlap": round(overlap_ratio, 3),
        "visualization_b64": visualization_b64,
    }


def null_model_permutation_test(
    enhanced_gray: np.ndarray,
    attention_map_gray: np.ndarray,
    num_permutations: int = 100,
    block_size: int = 16,
) -> dict:
    """Statistical null-model sanity test.

    Permutes attention spatial blocks to test the null hypothesis that the observed
    OAAS correlation arose by chance rather than learned anatomical alignment.
    """
    observed_result = orientation_attention_alignment(enhanced_gray, attention_map_gray, block_size)
    observed_r = observed_result["alignment_correlation"]

    fields = compute_orientation_field(enhanced_gray, block_size=block_size)
    singularity = singularity_map(fields["coherence"])
    flat_s = singularity.flatten()

    attention_resized = cv2.resize(
        attention_map_gray.astype(np.float32),
        (singularity.shape[1], singularity.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    a_lo, a_hi = float(attention_resized.min()), float(attention_resized.max())
    attention_norm = (attention_resized - a_lo) / (a_hi - a_lo + 1e-6)

    permuted_correlations = []
    rng = np.random.default_rng(seed=42)

    for _ in range(num_permutations):
        # Spatial permutation of attention tiles
        shuffled = rng.permutation(attention_norm.flatten())
        if shuffled.std() < 1e-6 or flat_s.std() < 1e-6:
            r_perm = 0.0
        else:
            r_perm = float(np.corrcoef(shuffled, flat_s)[0, 1])
        permuted_correlations.append(r_perm)

    p_value = float(np.mean(np.array(permuted_correlations) >= observed_r))

    return {
        "observed_correlation": observed_r,
        "null_mean_correlation": round(float(np.mean(permuted_correlations)), 3),
        "null_p_value": round(p_value, 4),
        "is_statistically_significant": bool(p_value < 0.05),
    }
