"""FastAPI server for facial emotion recognition via DeepFace."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Emotion Recognition API", version="2.0.0")

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

# ---------------------------------------------------------------------------
# HSEmotion singleton (lazy-loaded on first request)
# ---------------------------------------------------------------------------
_hsemotion_recognizer = None

HSEMOTION_LABELS = [
    "angry", "contempt", "disgust", "fear",
    "happy", "neutral", "sad", "surprise",
]


def _get_hsemotion():
    global _hsemotion_recognizer
    if _hsemotion_recognizer is None:
        from hsemotion.facial_emotions import HSEmotionRecognizer
        _hsemotion_recognizer = HSEmotionRecognizer(
            model_name="enet_b0_8_best_afew", device="cpu",
        )
        logger.info("HSEmotion model loaded.")
    return _hsemotion_recognizer


# ---------------------------------------------------------------------------
# Fine-tuned model singleton (lazy-loaded, optional)
# ---------------------------------------------------------------------------
_finetuned_model = None
_finetuned_labels: list[str] = []

MODELS_DIR = Path(__file__).resolve().parent / "models"
FINETUNED_WEIGHTS = MODELS_DIR / "emotion_efficientnet_b0.pth"
FINETUNED_LABELS_FILE = MODELS_DIR / "emotion_labels.json"


def _get_finetuned():
    global _finetuned_model, _finetuned_labels

    if _finetuned_model is not None:
        return _finetuned_model, _finetuned_labels

    if not FINETUNED_WEIGHTS.exists():
        raise HTTPException(
            status_code=404,
            detail="Fine-tuned model not found. Run the Colab notebook first and place the .pth in backend/models/.",
        )

    import timm
    import torch

    if FINETUNED_LABELS_FILE.exists():
        _finetuned_labels = json.loads(FINETUNED_LABELS_FILE.read_text())
    else:
        _finetuned_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    num_classes = len(_finetuned_labels)
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    state = torch.load(str(FINETUNED_WEIGHTS), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _finetuned_model = model
    logger.info("Fine-tuned model loaded (%d classes).", num_classes)
    return _finetuned_model, _finetuned_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _extract_face_crop(img: np.ndarray) -> tuple[np.ndarray, dict | None]:
    """Use DeepFace/RetinaFace to detect and crop the first face."""
    try:
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="retinaface",
            enforce_detection=True,
        )
    except Exception:
        faces = DeepFace.extract_faces(
            img_path=img,
            detector_backend="retinaface",
            enforce_detection=False,
        )

    if not faces:
        raise HTTPException(status_code=422, detail="No face detected.")

    face_obj = faces[0]
    face_pixels = face_obj.get("face")
    region = face_obj.get("facial_area")

    if face_pixels is None:
        raise HTTPException(status_code=422, detail="Face crop failed.")

    if face_pixels.max() <= 1.0:
        face_pixels = (face_pixels * 255).astype(np.uint8)

    return face_pixels, region


def _predict_hsemotion(img: np.ndarray) -> dict[str, Any]:
    face_pixels, region = _extract_face_crop(img)

    face_bgr = cv2.cvtColor(face_pixels, cv2.COLOR_RGB2BGR) if face_pixels.shape[-1] == 3 else face_pixels

    fer = _get_hsemotion()
    emotion_label, raw_scores = fer.predict_emotions(face_bgr, logits=True)

    import torch
    probs = torch.softmax(torch.tensor(raw_scores), dim=0).tolist()

    scores = {}
    for i, label in enumerate(HSEMOTION_LABELS):
        if i < len(probs):
            scores[label] = float(probs[i])

    dominant = max(scores, key=lambda k: scores[k])
    return {"dominant": dominant, "scores": scores, "region": region}


def _predict_finetuned(img: np.ndarray) -> dict[str, Any]:
    import torch
    import torchvision.transforms as T

    model, labels = _get_finetuned()
    face_pixels, region = _extract_face_crop(img)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(face_pixels).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

    scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
    dominant = max(scores, key=lambda k: scores[k])
    return {"dominant": dominant, "scores": scores, "region": region}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

ModelName = Literal["deepface", "hsemotion", "finetuned"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: ModelName = Query("deepface"),
) -> dict[str, Any]:
    """
    Accept a JPEG/PNG frame with a face; return dominant emotion and score map.
    Use ?model=deepface|hsemotion|finetuned to choose the inference engine.
    """
    contents = await file.read()
    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="Image too small or empty.")

    img = _decode_image(contents)

    if model == "hsemotion":
        return _predict_hsemotion(img)

    if model == "finetuned":
        return _predict_finetuned(img)

    # Default: DeepFace
    try:
        raw = DeepFace.analyze(
            img_path=img,
            actions=["emotion"],
            enforce_detection=True,
            detector_backend="retinaface",
        )
    except Exception as exc:
        logger.info("DeepFace analyze failed: %s", exc)
        try:
            raw = DeepFace.analyze(
                img_path=img,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="retinaface",
            )
        except Exception as exc2:
            logger.warning("Retry without enforce_detection failed: %s", exc2)
            raise HTTPException(
                status_code=422,
                detail="No clear face found. Try better lighting or move closer.",
            ) from exc2

    return _normalize_deepface_result(raw)
