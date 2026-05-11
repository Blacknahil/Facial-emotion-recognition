# Backend — Emotion recognition API

FastAPI service that accepts an image and returns **dominant emotion** and **per-class scores** using [DeepFace](https://github.com/serengil/deepface) (`emotion` analysis).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### TensorFlow / Keras note

This stack pulls **TensorFlow** and related deps. With TF **2.21+**, RetinaFace (face detection inside DeepFace) expects the **`tf-keras`** package; it is listed in `requirements.txt`. If imports fail mentioning `tf_keras`, run:

```bash
pip install tf-keras
```

## Run

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | `{ "status": "ok" }` |
| `POST` | `/predict` | Multipart form field **`file`** (JPEG/PNG image with a visible face) |

### Example response (`POST /predict`)

```json
{
  "dominant": "happy",
  "scores": {
    "angry": 0.01,
    "disgust": 0.0,
    "fear": 0.02,
    "happy": 0.72,
    "sad": 0.05,
    "surprise": 0.15,
    "neutral": 0.05
  },
  "region": { ... }
}
```

Scores are normalized to **0–1** (DeepFace sometimes returns 0–100; the API normalizes).

### Errors

- **400** — Bad or empty image bytes  
- **422** — No face detected or analysis failed (try lighting, face size, or `enforce_detection` behavior in code)

## CORS

Allowed origins include `http://localhost:5173` and `http://127.0.0.1:5173` for the Vite dev server. Extend `allow_origins` in `main.py` if you deploy elsewhere.

## Project files

| File | Role |
|------|------|
| `main.py` | FastAPI app, DeepFace pipeline |
| `requirements.txt` | Python dependencies |
