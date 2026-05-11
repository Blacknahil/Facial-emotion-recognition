# Facial emotion recognition (course demo)

Full-stack demo for a cognitive science project: **emotion classification from facial expressions** using **DeepFace** (machine learning) behind a **FastAPI** API and a **React + TypeScript** web UI with webcam and image upload.

## Repository layout

| Path | Description |
|------|-------------|
| [`backend/`](backend/) | REST API (`POST /predict`), DeepFace inference |
| [`frontend/`](frontend/) | Vite + React UI, proxies `/api` to the backend in development |

## Prerequisites

- **Python** 3.10+ (3.12 recommended)
- **Node.js** 18+ and npm
- Webcam optional (you can use image upload instead)

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The dev server forwards `/api/*` to the backend on port **8000**.

## First run notes

- **DeepFace** downloads model weights on first inference; that can take several minutes and uses noticeable disk space.
- If you see an error about **`tf_keras`** / **`tf-keras`**, ensure `pip install -r requirements.txt` completed (TensorFlow 2.21+ needs the `tf-keras` package—already listed in `requirements.txt`).

## Production build (frontend only)

```bash
cd frontend
npm run build
npm run preview    # serves dist/ — keep the API running on port 8000
```

## Cognitive science angle (report)

You can frame the write-up around **categorization** of affect from faces, **recognition** from prior exposure (memory for faces/expressions), and **ML as a model** of perception—with discussion of label ambiguity, lighting, and bias.

## License

Use as needed for your course submission; cite DeepFace and datasets if your instructor requires references.
