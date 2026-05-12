# Facial emotion recognition (course demo)

Full-stack demo for a cognitive science project: **emotion classification from facial expressions** using multiple ML models behind a **FastAPI** API and a **React + TypeScript** web UI with webcam and image upload.

## Repository layout

| Path | Description |
|------|-------------|
| [`backend/`](backend/) | REST API (`POST /predict`), DeepFace / HSEmotion / fine-tuned inference |
| [`frontend/`](frontend/) | Vite + React UI, proxies `/api` to the backend in development |
| [`notebooks/`](notebooks/) | Colab notebook for fine-tuning EfficientNet-B0 on FER+ |

## Models

The web UI lets you switch between three models at inference time:

| Model | What it is | Trained on |
|-------|-----------|------------|
| **DeepFace (baseline)** | Small CNN shipped with DeepFace, uses RetinaFace detector | FER-2013 (~36k images, noisy labels) |
| **HSEmotion (AffectNet)** | EfficientNet-B0 pretrained by HSE University | AffectNet (~400k images, 8 classes) |
| **Fine-tuned (your model)** | EfficientNet-B0 you fine-tune yourself in Colab | FER+ (Microsoft's relabeled FER-2013 with crowd-sourced soft labels) |

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

Open [http://localhost:5173](http://localhost:5173). The dev server forwards `/api/*` to the backend on port **8000**. Choose a model from the dropdown, then capture or upload a face.

## Fine-tuning your own model (Colab)

1. Open [`notebooks/finetune_fer.ipynb`](notebooks/finetune_fer.ipynb) in Google Colab.
2. Set the runtime to **GPU** (T4 is fine).
3. Upload `fer2013.csv` from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) when prompted.
4. Run all cells. Training takes ~20–30 minutes on a T4.
5. Download the two output files:
   - `emotion_efficientnet_b0.pth` — trained weights
   - `emotion_labels.json` — class-index-to-label mapping
6. Place both files in `backend/models/`.
7. Select **Fine-tuned (your model)** in the web UI dropdown.

## First run notes

- **DeepFace** and **HSEmotion** both download model weights on first inference; that can take a few minutes.
- If you see an error about **`tf_keras`** / **`tf-keras`**, ensure `pip install -r requirements.txt` completed (TensorFlow 2.21+ needs the `tf-keras` package—already listed in `requirements.txt`).
- The **fine-tuned** option returns a 404 until you run the Colab notebook and place the `.pth` file in `backend/models/`.

## Production build (frontend only)

```bash
cd frontend
npm run build
npm run preview    # serves dist/ — keep the API running on port 8000
```

## Cognitive science angle (report)

You can frame the write-up around **categorization** of affect from faces, **recognition** from prior exposure (memory for faces/expressions), and **ML as a model** of perception—with discussion of label ambiguity, lighting, and bias. The three-model comparison gives concrete evidence for how training data quality and model capacity affect classification.

## License


