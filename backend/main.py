from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.core.config import settings

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up and preload model on startup so user requests don't suffer initial delay
    app.state.model_ready = False
    try:
        from backend.ml.inference.predictor import ridgevision_predictor

        model_ready = (
            ridgevision_predictor._load_ensemble_models()
            or ridgevision_predictor._load_trained_model()
        )
        app.state.model_ready = model_ready
        if not model_ready:
            print(
                "Startup model warning: no trained model was found. "
                "Add model weights before using prediction in deployment."
            )
    except Exception as exc:
        print(f"Startup model warning: {exc}")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Experimental dermatoglyphic analysis API for research workflows.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins) or ["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", include_in_schema=False)
async def frontend_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
