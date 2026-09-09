import base64
from io import BytesIO
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image


def compute_attention_intensity(enhanced: np.ndarray) -> np.ndarray:
    """Raw grayscale fallback attention intensity map (0-255).

    Used when no trained neural network is loaded or during heuristic inference.
    """
    edges_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=5)
    edges_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=5)
    attention = cv2.magnitude(edges_x, edges_y)
    attention = cv2.GaussianBlur(attention, (0, 0), 3)
    attention = cv2.normalize(attention, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return attention


def encode_bgr_to_data_url(image_bgr: np.ndarray) -> str:
    """Shared PNG/base64 encoder across all explainability tiers."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    buffer = BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def compute_gradcam_plus_plus(
    model,
    image_batch: np.ndarray,
    texture_batch: Optional[np.ndarray] = None,
    class_index: Optional[int] = None,
    target_layer_name: Optional[str] = None,
) -> tuple[np.ndarray, str]:
    """Computes true Grad-CAM++ saliency via higher-order backpropagation.

    Grad-CAM++ weights positive partial derivatives with second and third-order
    gradients, providing superior spatial localization for fine fingerprint ridge detail.

    Returns:
        tuple[np.ndarray, str]: Normalized intensity map (H, W) in range [0, 255],
        and extraction method ('gradcam_plus_plus' or 'heuristic_fallback').
    """
    try:
        import tensorflow as tf
    except ImportError:
        # Fallback to edge magnitude if TensorFlow is unavailable
        gray = image_batch[0, :, :, 0] if image_batch.ndim == 4 else image_batch
        return compute_attention_intensity((gray * 255).astype(np.uint8)), "heuristic_fallback"

    # Identify target convolutional layer
    if target_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                target_layer_name = layer.name
                break
            # Handle nested models (e.g. EfficientNet application as a sub-layer)
            if hasattr(layer, "layers"):
                for sub in reversed(layer.layers):
                    if isinstance(sub, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                        target_layer_name = sub.name
                        break
                if target_layer_name:
                    break

    if target_layer_name is None:
        target_layer_name = "top_conv"

    try:
        # Construct gradient model tapping the target feature map and class logits
        target_layer = None
        for layer in model.layers:
            if layer.name == target_layer_name:
                target_layer = layer
                break
            if hasattr(layer, "layers"):
                for sub in layer.layers:
                    if sub.name == target_layer_name:
                        target_layer = sub
                        break

        if target_layer is None:
            # Fallback if specific layer not resolved
            enhanced_gray = (image_batch[0, :, :, 0] * 255).astype(np.uint8)
            return compute_attention_intensity(enhanced_gray), "heuristic_fallback"

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output],
        )

        model_inputs = (
            {"image_input": image_batch, "texture_input": texture_batch}
            if texture_batch is not None and isinstance(model.input, (list, dict))
            else image_batch
        )

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(image_batch)
            conv_outputs, predictions = grad_model(model_inputs)
            if isinstance(predictions, dict):
                preds_tensor = predictions.get("blood_group", predictions.get("logits", list(predictions.values())[0]))
            elif isinstance(predictions, (list, tuple)):
                preds_tensor = predictions[0]
            else:
                preds_tensor = predictions

            if class_index is None:
                class_index = int(tf.argmax(preds_tensor[0]))
            class_score = preds_tensor[:, class_index]

        # 1st, 2nd, and 3rd order gradients for Grad-CAM++
        grads_1 = tape.gradient(class_score, conv_outputs)
        grads_2 = tape.gradient(grads_1, conv_outputs)
        grads_3 = tape.gradient(grads_2, conv_outputs)
        del tape

        if grads_1 is None:
            enhanced_gray = (image_batch[0, :, :, 0] * 255).astype(np.uint8)
            return compute_attention_intensity(enhanced_gray), "heuristic_fallback"

        # Alpha weight computation
        conv_out = conv_outputs[0]
        g1 = grads_1[0]
        g2 = grads_2[0] if grads_2 is not None else tf.square(g1)
        g3 = grads_3[0] if grads_3 is not None else tf.pow(g1, 3)

        denominator = 2.0 * g2 + tf.reduce_sum(conv_out * g3, axis=(0, 1), keepdims=True)
        denominator = tf.where(denominator != 0.0, denominator, tf.ones_like(denominator))
        alphas = g2 / denominator

        relu_grads = tf.maximum(g1, 0.0)
        weights = tf.reduce_sum(alphas * relu_grads, axis=(0, 1))

        cam = tf.reduce_sum(weights * conv_out, axis=-1)
        cam = tf.maximum(cam, 0.0)

        cam_np = cam.numpy()
        h, w = image_batch.shape[1], image_batch.shape[2]
        cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)

        c_min, c_max = float(cam_resized.min()), float(cam_resized.max())
        if c_max > c_min:
            cam_norm = ((cam_resized - c_min) / (c_max - c_min) * 255.0).astype(np.uint8)
        else:
            cam_norm = np.zeros((h, w), dtype=np.uint8)

        return cam_norm, "gradcam_plus_plus"

    except Exception:
        # Failsafe fallback
        enhanced_gray = (image_batch[0, :, :, 0] * 255).astype(np.uint8)
        return compute_attention_intensity(enhanced_gray), "heuristic_fallback"


def heuristic_grad_cam_b64(
    original_bgr: np.ndarray,
    enhanced: np.ndarray,
    model=None,
    image_tensor: Optional[np.ndarray] = None,
    class_index: Optional[int] = None,
) -> str:
    """Generates the Tier 1 attention heatmap overlay.

    If a trained model and tensor are provided, executes true Grad-CAM++.
    Otherwise, applies smooth Gabor/Sobel edge attention mapping.
    """
    if model is not None and image_tensor is not None:
        attention, _ = compute_gradcam_plus_plus(model, image_tensor, class_index=class_index)
    else:
        attention = compute_attention_intensity(enhanced)

    heatmap = cv2.applyColorMap(attention, cv2.COLORMAP_TURBO)
    base = cv2.resize(original_bgr, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_AREA)
    overlay = cv2.addWeighted(base, 0.58, heatmap, 0.42, 0)
    return encode_bgr_to_data_url(overlay)
