"""FastAPI server for facial emotion recognition via DeepFace."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Emotion Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decode_image(contents: bytes) -> np.ndarray:
    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image data.")
    return img


def _normalize_deepface_result(raw: Any) -> dict[str, Any]:
    """DeepFace returns a dict or list of dicts depending on version/input."""
    if isinstance(raw, list):
        if not raw:
            raise HTTPException(status_code=422, detail="No face detected.")
        entry = raw[0]
    else:
        entry = raw

    emotion = entry.get("emotion")
    if not isinstance(emotion, dict):
        raise HTTPException(status_code=500, detail="Unexpected model output.")

    dominant_emotion = entry.get("dominant_emotion")
    if not dominant_emotion:
        dominant_emotion = max(emotion, key=lambda k: float(emotion[k]))

    raw_scores = {str(k): float(v) for k, v in emotion.items()}
    # DeepFace often returns 0–100; normalize to 0–1 for the API.
    max_val = max(raw_scores.values()) if raw_scores else 0.0
    if max_val > 1.0:
        scores = {k: v / 100.0 for k, v in raw_scores.items()}
    else:
        scores = raw_scores
    return {
        "dominant": str(dominant_emotion),
        "scores": scores,
        "region": entry.get("region"),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Accept a JPEG/PNG frame with a face; return dominant emotion and score map.
    """
    contents = await file.read()
    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="Image too small or empty.")

    img = _decode_image(contents)

    try:
        raw = DeepFace.analyze(
            img_path=img,
            actions=["emotion"],
            enforce_detection=True,
        )
    except Exception as exc:
        logger.info("DeepFace analyze failed: %s", exc)
        try:
            raw = DeepFace.analyze(
                img_path=img,
                actions=["emotion"],
                enforce_detection=False,
            )
        except Exception as exc2:
            logger.warning("Retry without enforce_detection failed: %s", exc2)
            raise HTTPException(
                status_code=422,
                detail="No clear face found. Try better lighting or move closer.",
            ) from exc2

    return _normalize_deepface_result(raw)
