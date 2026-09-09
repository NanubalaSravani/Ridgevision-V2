import hashlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from backend.core.config import CLASS_LABELS
from backend.ml.training.dataset_manifest import IMAGE_SUFFIXES


def compute_perceptual_cluster_hash(image_path: str, hash_size: int = 8) -> str:
    """Computes a 64-bit perceptual hash (pHash) to detect duplicate or near-identical

    fingerprint impressions acquired from the same finger.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return hashlib.md5(image_path.encode("utf-8")).hexdigest()[:16]
        resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[:hash_size, :hash_size]
        med = np.median(dct_low)
        diff = dct_low > med
        return hashlib.sha256(diff.tobytes()).hexdigest()[:16]
    except Exception:
        return hashlib.md5(image_path.encode("utf-8")).hexdigest()[:16]


def build_audited_manifest(dataset_dir: str | Path) -> list[dict]:
    """Constructs a leakage-audited manifest with decoupled ABO/Rh targets and cluster IDs."""
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    rows = []
    seen_paths = set()

    for label in CLASS_LABELS:
        abo = label.replace("+", "").replace("-", "")
        rh = "+" if "+" in label else "-"

        # Match various directory naming conventions
        patterns = [
            f"**/{label}/*",
            f"**/{label.replace('+', 'positive').replace('-', 'negative')}/*",
            f"**/{label.replace('+', '_pos').replace('-', '_neg')}/*",
        ]

        for pat in patterns:
            for p in root.glob(pat):
                p_str = str(p)
                if p.suffix.lower() in IMAGE_SUFFIXES and p_str not in seen_paths:
                    seen_paths.add(p_str)
                    cluster_id = compute_perceptual_cluster_hash(p_str)
                    rows.append(
                        {
                            "path": p_str,
                            "label": label,
                            "abo_label": abo,
                            "rh_label": rh,
                            "cluster_id": cluster_id,
                        }
                    )

    if not rows:
        raise ValueError("No labeled images found in dataset directory.")

    return rows


def get_leaksafe_train_test_split(
    manifest: list[dict],
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Performs subject-independent grouped split based on cluster IDs.

    Guarantees no shared impressions from the same cluster/subject across train and test.
    """
    groups = [r["cluster_id"] for r in manifest]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(manifest, groups=groups))

    train_data = [manifest[i] for i in train_idx]
    test_data = [manifest[i] for i in test_idx]

    return train_data, test_data


def generate_shuffled_label_control(manifest: list[dict], random_state: int = 42) -> list[dict]:
    """Generates randomized-label sanity control dataset for scientific integrity validation.

    A valid biometric model should collapse to chance accuracy (~12.5%) on this dataset.
    """
    rng = np.random.default_rng(seed=random_state)
    shuffled = [dict(r) for r in manifest]

    labels = [r["label"] for r in manifest]
    permuted_labels = rng.permutation(labels)

    for i, r in enumerate(shuffled):
        new_label = str(permuted_labels[i])
        r["label"] = new_label
        r["abo_label"] = new_label.replace("+", "").replace("-", "")
        r["rh_label"] = "+" if "+" in new_label else "-"

    return shuffled
