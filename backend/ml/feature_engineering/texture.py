import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy


class UnifiedTextureExtractor:
    """Unified biometric texture extractor reconciling UI reporting features and model input vectors.

    Eliminates discrepancies between user-facing feature dictionaries and downstream
    fusion tensor vectors by calculating them from a single canonical pass.
    """

    @staticmethod
    def _compute_core_metrics(enhanced: np.ndarray) -> dict:
        enhanced_224 = cv2.resize(enhanced, (224, 224), interpolation=cv2.INTER_AREA)

        # LBP (P=8, R=1, uniform)
        lbp_8 = local_binary_pattern(enhanced_224, P=8, R=1, method="uniform")
        lbp_hist, _ = np.histogram(
            lbp_8.ravel(),
            bins=np.arange(0, 11),
            range=(0, 10),
            density=True,
        )
        lbp_hist = np.nan_to_num(lbp_hist)

        # GLCM (4 directions: 0, 45, 90, 135 deg)
        glcm = graycomatrix(
            enhanced_224,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )

        glcm_props = {}
        glcm_vector_components = []
        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
            values = graycoprops(glcm, prop)[0]
            glcm_props[prop] = float(np.mean(values))
            glcm_vector_components.append(np.mean(values))
            glcm_vector_components.append(np.std(values))

        # Ridge and edge properties
        edges = cv2.Canny(enhanced_224, 50, 150)
        canny_ridge_density = float(np.mean(edges > 0))

        gx = cv2.Sobel(enhanced_224, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(enhanced_224, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)

        mean_mag = float(np.mean(magnitude) / 255.0)
        mean_intensity = float(np.mean(enhanced_224) / 255.0)
        std_intensity = float(np.std(enhanced_224) / 255.0)
        entropy_val = float(shannon_entropy(enhanced_224) / 8.0)
        otsu_ratio = float(np.mean(cv2.threshold(enhanced_224, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0))
        orientation_coherence = float(np.abs(np.mean(np.exp(1j * orientation))))
        laplacian_var = float(cv2.Laplacian(enhanced_224, cv2.CV_64F).var() / 10000.0)

        ridge_vector = np.array(
            [
                canny_ridge_density,
                mean_mag,
                mean_intensity,
                std_intensity,
                entropy_val,
                otsu_ratio,
                orientation_coherence,
                laplacian_var,
            ],
            dtype=np.float32,
        )

        # Construct 30-dim canonical fusion vector
        fusion_vector_30 = np.concatenate(
            [
                lbp_hist,
                np.array(glcm_vector_components, dtype=np.float32),
                ridge_vector,
            ]
        )
        fusion_vector_30 = np.nan_to_num(fusion_vector_30).astype(np.float32)

        # Construct consistent reporting dictionary
        feature_dict = {
            "lbp_uniformity": float(np.sum(lbp_hist**2)),
            "lbp_peak": float(np.max(lbp_hist)),
            "glcm_contrast": glcm_props["contrast"],
            "glcm_homogeneity": glcm_props["homogeneity"],
            "glcm_energy": glcm_props["energy"],
            "glcm_correlation": glcm_props["correlation"],
            "ridge_density": canny_ridge_density,
            "ridge_strength": mean_mag,
            "intensity_entropy": entropy_val * 8.0,
            "intensity_mean": mean_intensity,
            "intensity_std": std_intensity,
            "orientation_coherence": orientation_coherence,
            "laplacian_variance": laplacian_var,
        }

        return {
            "dict": feature_dict,
            "vector_30": fusion_vector_30,
            "enhanced_224": enhanced_224,
        }

    @classmethod
    def extract_unified(cls, enhanced: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        """Returns both consistent feature dictionary and 30-dim fusion vector."""
        res = cls._compute_core_metrics(enhanced)
        return res["dict"], res["vector_30"]

    @classmethod
    def extract_vector(cls, enhanced: np.ndarray) -> np.ndarray:
        return cls._compute_core_metrics(enhanced)["vector_30"]

    @classmethod
    def extract_dict(cls, enhanced: np.ndarray) -> dict[str, float]:
        return cls._compute_core_metrics(enhanced)["dict"]

    @classmethod
    def extract_multilbp_vector(cls, enhanced: np.ndarray) -> np.ndarray:
        enhanced_300 = cv2.resize(enhanced, (300, 300), interpolation=cv2.INTER_AREA)

        lbp_features = []
        for p_val, radius in [(8, 1), (16, 2), (24, 3)]:
            lbp = local_binary_pattern(enhanced_300, P=p_val, R=radius, method="uniform")
            hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, p_val + 3),
                range=(0, p_val + 2),
                density=True,
            )
            lbp_features.append(np.nan_to_num(hist))

        glcm = graycomatrix(
            enhanced_300,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )

        glcm_features = []
        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
            values = graycoprops(glcm, prop)[0]
            glcm_features.append(np.mean(values))
            glcm_features.append(np.std(values))

        edges = cv2.Canny(enhanced_300, 50, 150)
        ridge_density = np.mean(edges > 0)

        gx = cv2.Sobel(enhanced_300, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(enhanced_300, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)

        ridge_features = np.array(
            [
                ridge_density,
                np.mean(magnitude) / 255.0,
                np.mean(enhanced_300) / 255.0,
                np.std(enhanced_300) / 255.0,
                shannon_entropy(enhanced_300) / 8.0,
                np.mean(cv2.threshold(enhanced_300, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0),
                np.abs(np.mean(np.exp(1j * orientation))),
                cv2.Laplacian(enhanced_300, cv2.CV_64F).var() / 10000.0,
            ],
            dtype=np.float32,
        )

        vector = np.concatenate(
            [
                np.concatenate(lbp_features),
                np.array(glcm_features, dtype=np.float32),
                ridge_features,
            ]
        )
        return np.nan_to_num(vector).astype(np.float32)


# Backward-compatible API functions wrapping the unified extractor
def extract_texture_features(enhanced: np.ndarray) -> dict[str, float]:
    return UnifiedTextureExtractor.extract_dict(enhanced)


def extract_fusion_texture_vector(enhanced: np.ndarray) -> np.ndarray:
    return UnifiedTextureExtractor.extract_vector(enhanced)


def extract_fusion_texture_vector_multilbp(enhanced: np.ndarray) -> np.ndarray:
    return UnifiedTextureExtractor.extract_multilbp_vector(enhanced)
