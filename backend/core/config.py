from dataclasses import dataclass


CLASS_LABELS = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]


@dataclass(frozen=True)
class Settings:
    app_name: str = "RidgeVision AI"
    version: str = "0.1.0"
    image_size: int = 224
    disclaimer: str = "Experimental research output only. Not for clinical use."


settings = Settings()
