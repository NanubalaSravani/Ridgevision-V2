from pathlib import Path

from backend.core.config import CLASS_LABELS


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def build_labeled_manifest(dataset_dir: str | Path) -> list[dict[str, str]]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    rows: list[dict[str, str]] = []
    for label in CLASS_LABELS:
        for image_path in root.glob(f"**/{label.replace('+', 'positive').replace('-', 'negative')}/*"):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append({"path": str(image_path), "label": label})
        for image_path in root.glob(f"**/{label}/*"):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append({"path": str(image_path), "label": label})

    if not rows:
        raise ValueError("No labeled fingerprint images found. Check the dataset folder naming.")
    return rows
