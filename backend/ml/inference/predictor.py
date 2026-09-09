import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from backend.core.config import CLASS_LABELS, settings
from backend.ml.explainability.attention_alignment import orientation_attention_alignment
from backend.ml.explainability.causal_attribution import minutiae_causal_attribution
from backend.ml.explainability.grad_cam import (
    compute_attention_intensity,
    compute_gradcam_plus_plus,
    heuristic_grad_cam_b64,
)
from backend.ml.feature_engineering.texture import (
    extract_fusion_texture_vector,
    extract_fusion_texture_vector_multilbp,
    extract_texture_features,
)
from backend.ml.preprocessing.fingerprint import decode_image, enhance_fingerprint
from backend.ml.uncertainty.conformal import SplitConformalPredictor


MODEL_OUTPUT_LABELS = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O-", "O+"]


class PredictorError(ValueError):
    pass


class ChannelAttention:
    pass


class SpatialAttention:
    pass


class CBAM:
    pass


class RidgeVisionPredictor:
    def __init__(self) -> None:
        self.model_path = Path("models/ridgevision_full_model.keras")
        self.model_88_path = Path("models/ridgevision_b0_88_style_best.keras")
        self.model_91_path = Path("models/ridgevision_b3_91_style_best.keras")
        self.model = None
        self.model_88 = None
        self.model_91 = None
        self.model_image_size = (224, 224)
        self.tensorflow = None
        self.is_fusion_model = False
        self.is_ensemble_model = False
        self.conformal = SplitConformalPredictor(alpha=0.10, class_labels=MODEL_OUTPUT_LABELS)
        self.conformal.q_hat = 0.85

    def _load_trained_model(self) -> bool:
        if self.model is None:
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise PredictorError("TensorFlow is required to use trained models.") from exc

            self.tensorflow = tf
            custom_objects = self._custom_objects(tf)

            loaded = False
            # Check for the primary working model from 90-accuracy.ipynb first
            keras_candidates = [
                Path("models/ridgevision_full_model.keras"),
                Path("models/ridgevision_model.keras"),
            ]
            for kp in keras_candidates:
                if kp.exists():
                    try:
                        self.model = tf.keras.models.load_model(
                            kp,
                            custom_objects=custom_objects,
                            safe_mode=False,
                            compile=False,
                        )
                        self.model_path = kp
                        loaded = True
                        print(f"Loaded trained Keras model: {kp}")
                        break
                    except Exception as e:
                        print(f"Could not load {kp}: {e}")

            # Fallback: Load v2 architecture with weights if Keras model not present
            if not loaded:
                weights_paths = [
                    Path("models/ridgevision_model.weights.h5"),
                    Path("ridgevisionnet_results/ridgevision_full_model.weights.h5"),
                    Path("models/ridgevision_model.h5"),
                ]
                for wp in weights_paths:
                    if wp.exists():
                        try:
                            from backend.ml.models.architecture import build_ridgevision_net

                            built_model = build_ridgevision_net()
                            built_model.load_weights(str(wp))
                            self.model = built_model
                            loaded = True
                            print(f"Loaded v2 model architecture with weights from {wp}")
                            break
                        except Exception as e:
                            print(f"Could not load weights from {wp}: {e}")

            if not loaded:
                return False

            input_shape = self.model.input_shape
            self.is_fusion_model = isinstance(input_shape, list) and len(input_shape) >= 2

            image_shape = input_shape[0] if self.is_fusion_model else input_shape

            if image_shape[1] and image_shape[2]:
                self.model_image_size = (int(image_shape[2]), int(image_shape[1]))

            self.model_texture_dim = None
            if self.is_fusion_model and input_shape[1][-1]:
                self.model_texture_dim = int(input_shape[1][-1])

            print("Loaded model:", self.model_path)
            print("Model input shape:", self.model.input_shape)
            print("Fusion model:", self.is_fusion_model)
            print("Image size:", self.model_image_size)
            print("Texture dim:", self.model_texture_dim)

        return True

    def _load_ensemble_models(self) -> bool:
        if not self.model_88_path.exists() or not self.model_91_path.exists():
            return False

        if self.model_88 is None or self.model_91 is None:
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise PredictorError(
                    "TensorFlow is required to use the trained ensemble .keras models."
                ) from exc

            self.tensorflow = tf
            custom_objects = self._custom_objects(tf)

            self.model_88 = tf.keras.models.load_model(
                self.model_88_path,
                custom_objects=custom_objects,
                compile=False,
            )
            self.model_91 = tf.keras.models.load_model(
                self.model_91_path,
                custom_objects=custom_objects,
                compile=False,
            )
            self.is_ensemble_model = True
            self.is_fusion_model = True

            print("Loaded ensemble model 88:", self.model_88_path)
            print("Loaded ensemble model 91:", self.model_91_path)

        return True

    def _custom_objects(self, tf):
        from backend.ml.models.architecture import (
            AdaptiveGatedFusion,
            CBAM,
            ChannelAttention,
            RidgeOrientationField,
            ROAM,
            SpatialAttention,
        )

        return {
            "CBAM": CBAM,
            "ChannelAttention": ChannelAttention,
            "SpatialAttention": SpatialAttention,
            "RidgeOrientationField": RidgeOrientationField,
            "ROAM": ROAM,
            "AdaptiveGatedFusion": AdaptiveGatedFusion,
        }

    def _probabilities_for_processed(self, image_bgr: np.ndarray, enhanced_gray: np.ndarray) -> dict[str, float]:
        """Shared branch logic (ensemble / trained / research mode) given an already
        preprocessed image. Used both by predict() and by explainability methods that
        need to re-run inference on perturbed copies of the image (e.g. minutiae
        causal ablation)."""
        if self._load_ensemble_models():
            return self._ensemble_model_probabilities(image_bgr=image_bgr, enhanced_gray=enhanced_gray)
        if self._load_trained_model():
            return self._trained_model_probabilities(image_bgr=image_bgr, enhanced_gray=enhanced_gray)

        features = extract_texture_features(enhanced_gray)
        # research mode scores from features + a hash of pixel bytes; re-derive bytes
        # from the (possibly perturbed) image so the ablation actually changes the score
        image_bytes = cv2.imencode(".png", image_bgr)[1].tobytes()
        return self._research_mode_probabilities(features, image_bytes)

    def probabilities_for_bgr_image(self, image_bgr: np.ndarray) -> dict[str, float]:
        """Public entry point: full preprocessing + inference for a raw BGR image
        array (as opposed to `predict`, which takes raw upload bytes and also builds
        the user-facing response payload). Used by explainability modules that need
        to re-run inference on perturbed/ablated copies of the fingerprint."""
        processed = enhance_fingerprint(image_bgr)
        return self._probabilities_for_processed(image_bgr, processed["enhanced"])

    def predict(self, image_bytes: bytes, explain: bool = False) -> dict:
        try:
            image = decode_image(image_bytes)
            processed = enhance_fingerprint(image)
            features = extract_texture_features(processed["enhanced"])
        except ValueError as exc:
            raise PredictorError(str(exc)) from exc

        probabilities = self._probabilities_for_processed(image, processed["enhanced"])

        if self.is_ensemble_model:
            inference_mode = "trained_ensemble"
        elif self.model is not None:
            inference_mode = "trained_model"
        else:
            inference_mode = "research_mode"

        predicted_class = max(probabilities, key=probabilities.get)

        active_model = self.model or self.model_88 or self.model_91
        if active_model is not None:
            class_idx = (
                MODEL_OUTPUT_LABELS.index(predicted_class)
                if predicted_class in MODEL_OUTPUT_LABELS
                else None
            )
            img_resized = cv2.resize(image, self.model_image_size, interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype("float32")
            if not self.is_fusion_model:
                img_rgb = img_rgb / 255.0
            img_tensor = np.expand_dims(img_rgb, axis=0)

            texture_tensor = None
            if self.is_fusion_model:
                trained_enhanced = self._enhance_for_trained_texture(image, (224, 224))
                texture_vector = extract_fusion_texture_vector(trained_enhanced)
                texture_tensor = np.expand_dims(texture_vector, axis=0)

            attention_intensity, saliency_method = compute_gradcam_plus_plus(
                active_model, img_tensor, texture_batch=texture_tensor, class_index=class_idx
            )
            heatmap = heuristic_grad_cam_b64(
                processed["original_bgr"],
                processed["enhanced"],
                model=active_model,
                image_tensor=img_tensor,
                class_index=class_idx,
            )
        else:
            saliency_method = "heuristic_fallback"
            attention_intensity = compute_attention_intensity(processed["enhanced"])
            heatmap = heuristic_grad_cam_b64(processed["original_bgr"], processed["enhanced"])

        probs_arr = np.array([probabilities.get(lbl, 0.0) / 100.0 for lbl in MODEL_OUTPUT_LABELS], dtype=np.float32)
        conformal_eval = self.conformal.evaluate_decision(probs_arr)

        abo_group = predicted_class.replace("+", "").replace("-", "") if predicted_class else "Unknown"
        rh_factor = "+" if "+" in predicted_class else "-"

        result = {
            "predicted_class": predicted_class,
            "confidence": round(probabilities[predicted_class], 2),
            "all_probabilities": probabilities,
            "abo_group": abo_group,
            "rh_factor": rh_factor,
            "conformal_prediction_set": conformal_eval["prediction_set"],
            "decision_status": conformal_eval["status"],
            "abstention_reason": conformal_eval["abstention_reason"],
            "coverage_guarantee": conformal_eval["coverage_guarantee"],
            "grad_cam_b64": heatmap,
            "saliency_method": saliency_method,
            "features": {key: round(value, 4) for key, value in features.items()},
            "inference_mode": inference_mode,
            "model_type": "ensemble_fusion_cbam_texture"
            if self.is_ensemble_model
            else "fusion_cbam_texture"
            if self.is_fusion_model
            else "single_image_cnn",
            "disclaimer": settings.disclaimer,
        }

        if explain:
            oaas = orientation_attention_alignment(processed["enhanced"], attention_intensity)
            mca = minutiae_causal_attribution(
                self,
                image_bgr=processed["original_bgr"],
                enhanced_gray=processed["enhanced"],
                predicted_class=predicted_class,
                baseline_confidence=probabilities[predicted_class],
            )

            # Top-level fields the frontend's tiered explainability panel reads
            # directly (tier1Img/tier2Img/tier3Img + the minutiae list).
            result["tier1_attention_b64"] = heatmap
            result["tier2_oaas_b64"] = oaas.get("visualization_b64")
            result["tier3_mca_b64"] = mca.get("visualization_b64")
            result["top_minutiae"] = mca.get("top_minutiae", [])

            result["explainability"] = {
                "orientation_attention_alignment": {
                    key: value for key, value in oaas.items() if key != "visualization_b64"
                },
                "minutiae_causal_attribution": {
                    key: value for key, value in mca.items() if key != "visualization_b64"
                },
            }

        return result

    def _predict_fusion_model(
        self,
        model,
        image_bgr: np.ndarray,
        image_size: tuple[int, int],
        texture_vector: np.ndarray,
    ) -> np.ndarray:
        resized = cv2.resize(image_bgr, image_size, interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_batch = np.expand_dims(image_rgb.astype("float32"), axis=0)
        texture_batch = np.expand_dims(texture_vector, axis=0)

        return model.predict(
            {
                "image_input": image_batch,
                "texture_input": texture_batch,
            },
            verbose=0,
        )[0]

    def _enhance_for_trained_texture(self, image_bgr: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        responses = []
        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=4.0,
                theta=theta,
                lambd=10.0,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            responses.append(cv2.filter2D(gray, cv2.CV_32F, kernel))

        enhanced = np.max(np.stack(responses, axis=0), axis=0)
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        return enhanced.astype(np.uint8)

    def _ensemble_model_probabilities(self, image_bgr: np.ndarray, enhanced_gray: np.ndarray) -> dict[str, float]:
        enhanced_224 = self._enhance_for_trained_texture(image_bgr, (224, 224))
        enhanced_300 = self._enhance_for_trained_texture(image_bgr, (300, 300))

        predictions_88 = self._predict_fusion_model(
            self.model_88,
            image_bgr=image_bgr,
            image_size=(224, 224),
            texture_vector=extract_fusion_texture_vector(enhanced_224),
        )
        predictions_91 = self._predict_fusion_model(
            self.model_91,
            image_bgr=image_bgr,
            image_size=(300, 300),
            texture_vector=extract_fusion_texture_vector_multilbp(enhanced_300),
        )

        predictions = (0.5 * predictions_88) + (0.5 * predictions_91)

        return self._probability_dict(predictions, MODEL_OUTPUT_LABELS)

    def _trained_model_probabilities(self, image_bgr: np.ndarray, enhanced_gray: np.ndarray) -> dict[str, float]:
        resized = cv2.resize(image_bgr, self.model_image_size, interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if self.is_fusion_model:
            image_array = image_rgb.astype("float32")
        else:
            image_array = (image_rgb.astype("float32")) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)

        if self.is_fusion_model:
            if self.model_texture_dim == 74:
                trained_enhanced = self._enhance_for_trained_texture(image_bgr, (300, 300))
                texture_vector = extract_fusion_texture_vector_multilbp(trained_enhanced)
            else:
                trained_enhanced = self._enhance_for_trained_texture(image_bgr, (224, 224))
                texture_vector = extract_fusion_texture_vector(trained_enhanced)

            texture_batch = np.expand_dims(texture_vector, axis=0)

            raw_preds = self.model.predict(
                {
                    "image_input": image_batch,
                    "texture_input": texture_batch,
                },
                verbose=0,
            )
        else:
            raw_preds = self.model.predict(image_batch, verbose=0)

        if isinstance(raw_preds, dict):
            predictions = raw_preds["blood_group"][0]
        elif isinstance(raw_preds, (list, tuple)):
            predictions = raw_preds[0][0] if hasattr(raw_preds[0], "ndim") and raw_preds[0].ndim == 2 else raw_preds[0]
        else:
            predictions = raw_preds[0]

        return self._probability_dict(predictions, MODEL_OUTPUT_LABELS)

    def _research_mode_probabilities(self, features: dict[str, float], image_bytes: bytes) -> dict[str, float]:
        feature_vector = np.array(
            [
                features["lbp_uniformity"],
                features["lbp_peak"],
                features["glcm_contrast"],
                features["glcm_homogeneity"],
                features["glcm_energy"],
                features["ridge_density"],
                features["ridge_strength"],
                features["intensity_entropy"],
            ],
            dtype=np.float64,
        )

        seed = int(hashlib.sha256(image_bytes[:4096]).hexdigest()[:12], 16)

        scores = []
        for index, label in enumerate(CLASS_LABELS):
            phase = (seed % (997 + index * 13)) / 997.0
            weights = np.sin(np.arange(1, feature_vector.size + 1) * (index + 1.7 + phase))
            score = float(np.dot(feature_vector, weights))
            score += math.cos((index + 1) * features["ridge_density"] * 6.0)
            score += 0.18 if "+" in label and features["ridge_strength"] > 0.18 else 0
            scores.append(score)

        scores_array = np.array(scores, dtype=np.float64)
        scores_array = scores_array - scores_array.max()
        exp_scores = np.exp(scores_array)
        normalized = exp_scores / exp_scores.sum()

        return self._probability_dict(normalized, CLASS_LABELS)

    def _probability_dict(self, predictions: np.ndarray, labels: list[str]) -> dict[str, float]:
        return {
            label: round(float(probability * 100), 2)
            for label, probability in zip(labels, predictions)
        }


ridgevision_predictor = RidgeVisionPredictor()
