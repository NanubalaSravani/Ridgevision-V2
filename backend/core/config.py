import os
from dataclasses import dataclass, field


CLASS_LABELS = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]


@dataclass(frozen=True)
class Settings:
    app_name: str = "RidgeVision AI"
    version: str = "0.1.0"
    image_size: int = 224
    disclaimer: str = "Experimental research output only. Not for clinical use."
    allowed_origins: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_environment(cls) -> "Settings":
        origins = os.getenv("RIDGEVISION_ALLOWED_ORIGINS", "").strip()
        allowed_origins = tuple(item.strip() for item in origins.split(",") if item.strip())
        return cls(
            app_name=os.getenv("RIDGEVISION_APP_NAME", "RidgeVision AI"),
            version=os.getenv("RIDGEVISION_VERSION", "0.1.0"),
            image_size=int(os.getenv("RIDGEVISION_IMAGE_SIZE", "224")),
            disclaimer=os.getenv(
                "RIDGEVISION_DISCLAIMER",
                "Experimental research output only. Not for clinical use.",
            ),
            allowed_origins=allowed_origins,
        )


settings = Settings.from_environment()
