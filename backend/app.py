"""ВТОРник — server-side ONNX inference API.

POST /classify
  multipart/form-data: image=<JPEG/PNG, <=2MB>
  header X-API-Key: <token>
  -> { "class": "plastic", "confidence": 0.87, "probs": {"glass":..., ...} }
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

CLASSES = ["glass", "metal", "paper", "plastic"]
MODEL_PATH = Path(__file__).parent / "model.onnx"
MAX_IMAGE_BYTES = 2 * 1024 * 1024

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

app = FastAPI(title="ВТОРник classifier", version="1.0.0")
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
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return arr[np.newaxis, ...]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exps = np.exp(shifted)
    return exps / exps.sum()


@app.get("/")
def health():
    return {"status": "ok", "classes": CLASSES, "input": input_name}


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

    output = session.run(None, {input_name: tensor})[0][0]
    probs = softmax(output.astype(np.float64))
    idx = int(np.argmax(probs))

    return {
        "class": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probs": {cls: float(p) for cls, p in zip(CLASSES, probs)},
    }
