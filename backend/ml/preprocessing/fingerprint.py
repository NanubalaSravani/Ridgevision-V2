import cv2
import numpy as np

from backend.core.config import settings


def decode_image(image_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image. Use a valid PNG, JPG, or WEBP file.")
    return image


def enhance_fingerprint(image: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (settings.image_size, settings.image_size), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    contrast = clahe.apply(resized)
    denoised = cv2.GaussianBlur(contrast, (3, 3), 0)

    gabor_responses = []
    for theta in np.linspace(0, np.pi, 8, endpoint=False):
        kernel = cv2.getGaborKernel((17, 17), 4.0, theta, 10.0, 0.55, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(denoised, cv2.CV_32F, kernel)
        gabor_responses.append(filtered)

    enhanced = np.max(np.stack(gabor_responses, axis=0), axis=0)
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    normalized = enhanced.astype(np.float32) / 255.0

    return {
        "original_bgr": image,
        "gray": resized,
        "contrast": contrast,
        "enhanced": enhanced,
        "normalized": normalized,
    }
