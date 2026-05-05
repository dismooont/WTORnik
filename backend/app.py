"""ВТОРник — server-side ONNX inference API.

POST /classify
  multipart/form-data: image=<JPEG/PNG, <=2MB>
  header X-API-Key: <token>
  -> {
        "class": "plastic" | "glass" | "metal" | "paper" | "unknown",
        "confidence": 0.87,
        "detected_object": "water_bottle",
        "probs": {"glass":..., "metal":..., "paper":..., "plastic":...}
     }

Inference pipeline:
  JPEG -> PIL resize 224x224 -> NCHW (x/255 - mean) / std -> MobileNetV3-Large
  -> 1000 ImageNet logits -> softmax -> aggregate by IMAGENET_TO_WASTE
  -> if waste_total < UNKNOWN_THRESHOLD: "unknown" else argmax of 4 categories.
"""

import io
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from imagenet_classes import IMAGENET_CLASSES
from imagenet_to_waste import IMAGENET_TO_WASTE

WASTE_CATEGORIES = ("glass", "metal", "paper", "plastic")
UNKNOWN_THRESHOLD = 0.35

MODEL_PATH = Path(__file__).parent / "model.onnx"
MAX_IMAGE_BYTES = 2 * 1024 * 1024

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

API_KEY = os.environ.get("API_KEY", "dev-key-change-me")
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if o.strip()
]

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])

app = FastAPI(title="ВТОРник classifier", version="2.0.0")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})


def require_api_key(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def preprocess(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, ...]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exps = np.exp(shifted)
    return exps / exps.sum()


def aggregate(probs_1000: np.ndarray) -> tuple[str, float, str, dict[str, float]]:
    grouped = {c: 0.0 for c in WASTE_CATEGORIES}
    for idx, cat in IMAGENET_TO_WASTE.items():
        if cat in grouped:
            grouped[cat] += float(probs_1000[idx])

    waste_total = sum(grouped.values())
    top_idx = int(np.argmax(probs_1000))
    detected = IMAGENET_CLASSES[top_idx]

    if waste_total < UNKNOWN_THRESHOLD:
        return "unknown", float(waste_total), detected, grouped

    best = max(WASTE_CATEGORIES, key=lambda c: grouped[c])
    return best, grouped[best], detected, grouped


@app.get("/")
def health():
    return {
        "status": "ok",
        "classes": list(WASTE_CATEGORIES) + ["unknown"],
        "input": input_name,
        "model": "MobileNetV3-Large (ImageNet1K_V2)",
        "threshold": UNKNOWN_THRESHOLD,
    }


@app.post("/classify")
@limiter.limit("30/minute")
async def classify(
    request: Request,
    image: UploadFile = File(...),
    _: None = Depends(require_api_key),
):
    raw = await image.read()
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image too large (>2MB)")
    if not raw:
        raise HTTPException(status_code=400, detail="empty image")

    try:
        tensor = preprocess(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"cannot decode image: {exc}")

    logits = session.run(None, {input_name: tensor})[0][0]
    probs = softmax(logits.astype(np.float64))
    waste_class, confidence, detected_object, grouped = aggregate(probs)

    return {
        "class": waste_class,
        "confidence": confidence,
        "detected_object": detected_object,
        "probs": grouped,
    }
