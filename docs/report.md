# Facial Emotion Recognition Using Deep Learning: A Comparative Study

## Abstract

This report presents a full-stack facial emotion recognition system developed as part of a Cognitive Science course. The system classifies human facial expressions into seven emotion categories (angry, disgust, fear, happy, sad, surprise, neutral) using three distinct deep learning approaches: (1) a baseline DeepFace CNN trained on FER-2013, (2) an HSEmotion classifier pretrained on AffectNet, and (3) an EfficientNet-B0 model fine-tuned on the FER+ dataset. By comparing models that differ in architecture, training data, and label quality, this project investigates how these factors influence automated affect recognition — and draws parallels to human emotion perception as studied in cognitive science.

---

## 1. Introduction

Facial expressions are a primary channel through which humans communicate emotional states. The automatic recognition of these expressions has applications in human-computer interaction, mental health monitoring, and affective computing. However, building reliable emotion classifiers is difficult: expressions are subjective, context-dependent, and culturally variable.

This project implements a web-based emotion recognition system that allows users to capture facial images via webcam or file upload and receive real-time emotion predictions. To move beyond a single-model baseline, we compare three models with distinct training backgrounds, providing empirical evidence for how data quality and model architecture affect classification performance.

### 1.1 Objectives

1. Build a functional full-stack application for real-time facial emotion classification.
2. Evaluate a baseline model (DeepFace) and identify its limitations.
3. Integrate a stronger pretrained model (HSEmotion) for comparison.
4. Fine-tune a custom model on the FER+ dataset and assess improvement.
5. Discuss findings through the lens of cognitive science theories of emotion perception.

---

## 2. Background

### 2.1 Emotion Categorization Theories

**Basic emotions theory** (Ekman, 1971) proposes six universal emotions — happiness, sadness, fear, anger, disgust, and surprise — each associated with distinct facial muscle configurations (action units). This categorical framework underpins most facial expression recognition datasets and models, including all three used in this project.

An alternative is the **circumplex model of affect** (Russell, 1980), which places emotions along two continuous dimensions: **valence** (pleasant–unpleasant) and **arousal** (calm–excited). While our models output discrete labels, the probability distributions they produce over all classes can be interpreted as approximations of this continuous space.

### 2.2 Datasets

| Dataset | Images | Resolution | Classes | Labeling method | Known issues |
|---------|--------|-----------|---------|----------------|-------------|
| **FER-2013** | 35,887 | 48×48 grayscale | 7 | Single annotator | ~30% estimated mislabel rate, heavy neutral bias |
| **AffectNet** | ~400,000 | Variable (color) | 8 | Expert-annotated | Class imbalance, some ambiguous labels |
| **FER+** | 35,887 | 48×48 grayscale | 7 (+contempt, unknown, NF) | 10 crowd-sourced annotators per image | Inherits FER-2013 images; improved labels via majority vote |

FER+ (Barsoum et al., 2016) is a relabeling of FER-2013 by Microsoft Research. Each image was annotated by 10 independent taggers, and the majority-vote label replaces the original single-annotator label. Images where annotators could not agree (labeled "unknown" or "Not a Face") are excluded from training. This substantially reduces label noise.

### 2.3 EfficientNet Architecture

EfficientNet (Tan & Le, 2019) uses a compound scaling method that uniformly scales network width, depth, and resolution using a set of fixed scaling coefficients. EfficientNet-B0, the baseline variant, achieves strong performance relative to its parameter count (~5.3M parameters), making it suitable for transfer learning on moderately sized datasets like FER+.

---

## 3. System Architecture

### 3.1 Overview

The system follows a client-server architecture:

- **Frontend** — a React 18 + TypeScript single-page application built with Vite. It captures webcam frames or accepts image uploads, sends them to the backend via the Fetch API, and renders predictions including the dominant emotion label and a sorted score breakdown.
- **Backend** — a Python FastAPI server. It receives multipart image uploads, performs face detection using RetinaFace, routes to the selected inference engine, and returns a JSON response.

### 3.2 Request Flow

1. User selects a model from the dropdown (DeepFace / HSEmotion / Fine-tuned).
2. User captures a webcam frame or uploads an image.
3. Frontend sends the image as a JPEG blob via `POST /api/predict?model=<name>`.
4. Vite dev server proxies `/api/*` to the backend on port 8000.
5. Backend decodes the image, detects and crops the face with RetinaFace, runs the selected model.
6. Backend returns `{ dominant, scores, region }` as JSON.
7. Frontend displays the dominant emotion and score percentages.

### 3.3 API Specification

**Endpoint:** `POST /predict?model=<deepface|hsemotion|finetuned>`

**Request:** Multipart form data with field `file` containing a JPEG or PNG image.

**Response (200):**
```json
{
  "dominant": "happy",
  "scores": {
    "angry": 0.02,
    "disgust": 0.01,
    "fear": 0.03,
    "happy": 0.78,
    "sad": 0.05,
    "surprise": 0.08,
    "neutral": 0.03
  },
  "region": { "x": 120, "y": 80, "w": 200, "h": 200 }
}
```

**Error responses:** 400 (bad image), 404 (fine-tuned model file missing), 422 (no face detected), 500 (unexpected model output).

### 3.4 Face Detection

All three models share a common face detection step using **RetinaFace** (Deng et al., 2020). RetinaFace performs single-stage face localization with five facial landmark points (left eye, right eye, nose, left mouth corner, right mouth corner), producing tightly aligned face crops. This is a significant improvement over the default OpenCV Haar cascade detector, which produces loose bounding boxes that include background noise.

### 3.5 Technology Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | React 18, TypeScript, Vite 6, CSS |
| Backend | Python, FastAPI, Uvicorn, OpenCV |
| ML / Inference | DeepFace, HSEmotion, PyTorch, timm, TensorFlow/Keras |
| Face detection | RetinaFace (via DeepFace) |
| Fine-tuning | PyTorch, timm, Google Colab (T4 GPU) |

---

## 4. Models

### 4.1 Model 1 — DeepFace (Baseline)

- **Architecture:** Small custom CNN with convolutional and dense layers.
- **Training data:** FER-2013 (35,887 grayscale 48×48 images, 7 emotion classes).
- **Labels:** Single annotator per image with significant noise (~30% estimated error rate).
- **Reported accuracy:** ~60–65% on FER-2013 test set.
- **How it is loaded:** Weights are auto-downloaded from GitHub on first inference and cached locally in `~/.deepface/weights/`.

**Observed limitations:** During initial testing, this model exhibited a strong bias toward predicting "neutral" regardless of the user's actual expression. This is consistent with the known class imbalance in FER-2013 and the noise in its labels.

### 4.2 Model 2 — HSEmotion (AffectNet)

- **Architecture:** EfficientNet-B0 (~5.3M parameters).
- **Training data:** AffectNet (~400,000 facial images, 8 emotion classes including contempt).
- **Labels:** Expert-annotated with higher inter-rater agreement than FER-2013.
- **Reported accuracy:** ~63–66% on AffectNet validation (8-class).
- **How it is loaded:** Installed via the `hsemotion` Python package. Model weights are auto-downloaded on first inference.
- **Reference:** Savchenko (2021).

This model represents an off-the-shelf improvement: same EfficientNet-B0 architecture, but trained on a much larger and better-labeled dataset.

### 4.3 Model 3 — Fine-tuned EfficientNet-B0 (FER+)

- **Architecture:** EfficientNet-B0, pretrained on ImageNet (1.2M images, 1000 classes), fine-tuned for 7-class emotion classification.
- **Training data:** FER+ (Microsoft's relabeled FER-2013 with crowd-sourced soft labels from 10 annotators per image).
- **How it is loaded:** Trained weights (`emotion_efficientnet_b0.pth`) are stored in `backend/models/` and loaded directly from disk at startup. No download required.

This model represents our primary contribution: a custom-trained classifier where we control the training process, data preprocessing, and hyperparameters.

---

## 5. Methodology

### 5.1 Data Preparation

The training pipeline (implemented in `notebooks/finetune_fer.ipynb`) merges two CSV files:

1. **`fer2013.csv`** (from Kaggle) — contains the pixel data as space-separated integers (48×48 = 2,304 values per row) and dataset split labels (`Training`, `PublicTest`, `PrivateTest`).
2. **`fer2013new.csv`** (from Microsoft FER+ GitHub) — contains vote counts from 10 annotators across 10 columns: neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, NF.

The files are aligned row-by-row. We extract the 7 primary emotion vote columns, compute the majority-vote label via `argmax`, and discard rows where all emotion votes sum to zero (indicating "Not a Face" or unclassifiable images).

**Final dataset splits:**

| Split | Usage |
|-------|-------|
| Training | ~28,000 samples |
| Validation (PublicTest) | ~3,500 samples |
| Test (PrivateTest) | ~3,500 samples |

### 5.2 Preprocessing and Augmentation

Each 48×48 grayscale image is converted to RGB (by replicating the single channel) and resized to 224×224 to match EfficientNet-B0's expected input.

**Training augmentations:**
- Random horizontal flip (p=0.5)
- Random rotation (±10°)
- Color jitter (brightness ±20%, contrast ±20%)
- Normalization with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Validation/test preprocessing:**
- Resize to 224×224
- Normalization with ImageNet statistics (same as above)

### 5.3 Transfer Learning Strategy

We adopt a two-phase transfer learning approach:

**Phase 1 — Head-only training (5 epochs):**
- Freeze all backbone parameters (convolutional layers from ImageNet pretraining).
- Train only the new classification head (1,000-class ImageNet head replaced with a 7-class linear layer).
- Optimizer: Adam, learning rate 1e-3.
- Trainable parameters: 8,967.
- Purpose: learn a reasonable mapping from the frozen feature space to emotion classes without disturbing the pretrained representations.

**Phase 2 — Full fine-tuning (10 epochs):**
- Unfreeze all layers for end-to-end training.
- Optimizer: Adam, learning rate 1e-4, weight decay 1e-4.
- Learning rate schedule: cosine annealing over the full phase.
- Trainable parameters: ~5.3M (full network).
- Purpose: adapt the backbone's feature extraction to the specific characteristics of facial emotion images.

**Loss function:** Cross-entropy loss.
**Batch size:** 64.
**Hardware:** Google Colab with T4 GPU.

### 5.4 Model Checkpointing

The best model (highest validation accuracy) is saved during Phase 2. This ensures the final exported model corresponds to peak generalization performance rather than the last epoch (which may overfit).

---

## 6. Results

### 6.1 Fine-tuning Training Progress

**Phase 1 (frozen backbone, 5 epochs):**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 1 | 2.1435 | 36.19% | 1.9240 | 34.48% |
| 2 | 1.6496 | 44.41% | 1.7368 | 37.86% |
| 3–5 | Continued improvement | ~47% | — | ~45% |

With only 8,967 trainable parameters, Phase 1 establishes a reasonable baseline mapping from frozen ImageNet features to emotion labels. The limited capacity constrains performance.

**Phase 2 (full fine-tuning, 10 epochs):**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Saved |
|-------|-----------|-----------|----------|---------|-------|
| 1 | — | — | — | ~72% | |
| 5 | 0.2790 | 89.88% | 0.5267 | 82.01% | Best |
| 10 | — | ~94% | — | ~83% | |

Once the full network is unfrozen, accuracy improves dramatically. The best validation accuracy of **82.01%** was achieved at epoch 5 of Phase 2 (epoch 10 overall).

### 6.2 Model Comparison

| Model | Architecture | Training Data | Labels | Approx. Accuracy | Neutral Bias |
|-------|-------------|--------------|--------|-----------------|-------------|
| DeepFace (baseline) | Small CNN | FER-2013 (36k) | Single annotator | ~60–65% | Severe |
| HSEmotion | EfficientNet-B0 | AffectNet (400k) | Expert | ~63–66% (8-class) | Moderate |
| Fine-tuned (FER+) | EfficientNet-B0 | FER+ (28k usable) | 10 annotators (majority vote) | **~82%** (7-class) | Low |

### 6.3 Qualitative Observations

During live testing with the webcam interface:

- **DeepFace** frequently predicted "neutral" even for clearly happy or surprised expressions, consistent with its known bias.
- **HSEmotion** showed improved sensitivity to non-neutral emotions, though it occasionally confused similar emotions (e.g., sad vs. neutral, fear vs. surprise).
- **Fine-tuned (FER+)** demonstrated the most responsive behavior, correctly identifying happy, surprised, and angry expressions in most cases. Subtle expressions (mild sadness, mild fear) remained challenging.

---

## 7. Discussion

### 7.1 Data Quality vs. Model Architecture

The most striking finding is that the fine-tuned model achieved ~82% accuracy on FER+ despite using the **same images** as FER-2013 (which yields only ~60–65% for the DeepFace baseline). The only difference is the labels. This demonstrates that **label quality is at least as important as model architecture or dataset size** for this task.

HSEmotion, trained on a much larger dataset (400k vs. 36k images), achieves comparable or lower accuracy on its own benchmark. This suggests that for emotion recognition specifically, reducing label noise (as FER+ does through crowd-sourcing) provides more value than simply collecting more data with noisy labels.

### 7.2 Transfer Learning Effectiveness

The two-phase training strategy proved effective. Phase 1 (frozen backbone) rapidly learned a reasonable class mapping with minimal risk of catastrophic forgetting. Phase 2 (full fine-tuning) with a lower learning rate and cosine annealing allowed the backbone to adapt its feature extraction to facial emotion patterns while maintaining the structural knowledge from ImageNet pretraining.

The gap between training accuracy (~90%) and validation accuracy (~82%) at the best checkpoint indicates some overfitting, which is expected given the relatively small dataset size (28k training images). Further regularization (dropout, stronger augmentation, mixup) could potentially narrow this gap.

### 7.3 Limitations

1. **Static analysis** — The system analyzes single frames. Human emotion perception integrates temporal dynamics (how an expression changes over time), body language, voice, and contextual cues.
2. **Demographic bias** — FER-2013, AffectNet, and FER+ all under-represent certain ethnicities, age groups, and cultural expressions of emotion. Model performance may vary across demographics.
3. **Expression intensity** — All three models perform better on exaggerated expressions (wide smiles, raised eyebrows) than subtle ones. This contrasts with human perception, which is sensitive to micro-expressions.
4. **Categorical constraint** — The models force each face into a single dominant category. Real emotional states are often mixed (e.g., bittersweet = happy + sad) and better captured by dimensional models.
5. **Lighting and angle sensitivity** — Webcam conditions (backlighting, off-angle faces) degrade detection and classification quality.

---

## 8. Cognitive Science Perspective

### 8.1 Categorization of Affect

The models implement a **categorical model** of emotion, classifying faces into discrete labels. This mirrors the **basic emotions** theory (Ekman, 1971), which proposes that certain emotions are universally recognized from facial expressions and associated with distinct action unit configurations.

However, the dimensional model of affect (Russell, 1980) argues that emotions are better described along continuous axes of **valence** (pleasant–unpleasant) and **arousal** (calm–excited). The per-class probability scores returned by our models can be interpreted as a rough approximation of this continuous space — a face with high "happy" and moderate "surprise" scores maps to high valence and high arousal.

### 8.2 Label Ambiguity and Human Perception

The FER+ dataset directly illustrates the subjective nature of emotion perception. When 10 annotators label the same face, they frequently disagree — some images receive votes spread across three or four emotion categories. This mirrors findings in psychology that emotion recognition is:

- **Context-dependent** — the same facial expression is interpreted differently depending on the surrounding situation (Barrett et al., 2011).
- **Culturally variable** — display rules and recognition accuracy vary across cultures.
- **Perceiver-dependent** — influenced by the observer's own emotional state, personality, and experience.

### 8.3 ML as a Model of Perception

The three models offer an analogy to levels of perceptual expertise:

- **DeepFace (baseline)** — a naive perceiver with limited, noisy exposure. Like a child learning emotion categories from inconsistent feedback, it defaults to the most common category (neutral).
- **HSEmotion** — a more experienced perceiver trained on diverse, expert-curated examples. It has broader exposure but was not specifically calibrated for the task at hand.
- **Fine-tuned (FER+)** — a perceiver that has been calibrated using consensus-based feedback (crowd-sourced labels). Like an adult whose emotion recognition has been refined through social learning and repeated feedback from multiple people.

This progression demonstrates that training signal quality — analogous to the quality of social feedback in human development — is a critical determinant of perceptual accuracy.

### 8.4 Implications for Affective Computing

The gap between model performance and human emotion recognition highlights that current ML approaches capture surface-level statistical patterns (pixel distributions associated with labels) rather than genuine understanding of emotional states. True emotion recognition likely requires multimodal input (face, voice, body, context) and temporal reasoning — mirroring the integrated processing performed by the human brain.

---

## 9. Technical Setup and Reproducibility

### 9.1 Prerequisites

- Python 3.10+ (3.12 recommended)
- Node.js 18+ and npm
- Google Colab account (for fine-tuning; T4 GPU recommended)

### 9.2 Running the Application

**Backend:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

**Frontend (in a second terminal):**
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and select a model from the dropdown.

### 9.3 Fine-tuning (Optional — pretrained weights included)

The trained model weights (`emotion_efficientnet_b0.pth`) are included in the repository under `backend/models/`. To reproduce or modify the training:

1. Open `notebooks/finetune_fer.ipynb` in Google Colab.
2. Set runtime to GPU (T4).
3. The notebook mounts Google Drive to load `fer2013.csv`; alternatively, upload it manually when prompted.
4. FER+ labels (`fer2013new.csv`) are downloaded automatically from the Microsoft FER+ GitHub repository.
5. Run all cells (~20–30 minutes on T4).
6. Download the output files (`emotion_efficientnet_b0.pth` and `emotion_labels.json`) and place them in `backend/models/`.

### 9.4 Repository Structure

```
├── README.md                          Project overview and quick start
├── docs/
│   └── report.md                      This report
├── backend/
│   ├── main.py                        FastAPI server with 3 model endpoints
│   ├── requirements.txt               Python dependencies
│   ├── README.md                      API documentation
│   └── models/
│       ├── emotion_efficientnet_b0.pth   Fine-tuned model weights (16 MB)
│       └── emotion_labels.json           Class-index-to-label mapping
├── frontend/
│   ├── src/
│   │   ├── App.tsx                    Main UI with model selector
│   │   ├── App.css                    Styling
│   │   ├── main.tsx                   React entry point
│   │   └── index.css                  Global styles
│   ├── index.html                     HTML shell
│   ├── vite.config.ts                 Dev server + API proxy config
│   ├── package.json                   npm dependencies
│   └── tsconfig.json                  TypeScript configuration
└── notebooks/
    └── finetune_fer.ipynb             Colab notebook for fine-tuning
```

---

## 10. Conclusion

This project demonstrates that facial emotion recognition from static images is achievable with modern deep learning, but classification quality is highly dependent on training data characteristics. Our comparison of three models reveals that:

1. **Label quality matters more than dataset size.** The fine-tuned model achieved ~82% accuracy on FER+ (28k images with crowd-sourced labels) compared to ~60–65% for DeepFace trained on FER-2013 (the same images with single-annotator labels).

2. **Transfer learning is effective for emotion recognition.** Starting from ImageNet-pretrained features and fine-tuning on FER+ required only ~30 minutes of GPU training to surpass models trained on much larger datasets.

3. **Automated emotion recognition remains limited.** Even our best model operates on single-frame, face-only input with discrete categories — far from the multimodal, temporal, contextual processing that characterizes human emotion perception.

From a cognitive science standpoint, these models serve as useful but constrained computational analogies for categorical emotion perception. The FER+ annotation process itself — where 10 humans frequently disagree on the same face — is perhaps the most compelling illustration that emotion recognition is inherently subjective, even for humans.

---

## References

- Barrett, L. F., Mesquita, B., & Gendron, M. (2011). Context in emotion perception. *Current Directions in Psychological Science*, 20(5), 286–290.
- Barsoum, E., Zhang, C., Ferrer, C. C., & Zhang, Z. (2016). Training deep networks for facial expression recognition with crowd-sourced label distribution. *Proceedings of the 18th ACM International Conference on Multimodal Interaction (ICMI)*, 279–283.
- Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020). RetinaFace: Single-shot multi-level face localisation in the wild. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 5203–5212.
- Ekman, P. (1971). Universals and cultural differences in facial expressions of emotion. In J. Cole (Ed.), *Nebraska Symposium on Motivation* (Vol. 19, pp. 207–282). University of Nebraska Press.
- Goodfellow, I. J., Erhan, D., Carrier, P. L., Courville, A., Mirza, M., Hamner, B., ... & Bengio, Y. (2013). Challenges in representation learning: A report on three machine learning contests. *International Conference on Neural Information Processing (ICONIP)*, 117–124.
- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178.
- Savchenko, A. V. (2021). Facial expression and attributes recognition based on multi-task learning of lightweight neural networks. *IEEE International Symposium on Signal Processing and Information Technology (ISSPIT)*, 119–124.
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 6105–6114.
