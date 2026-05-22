"""Generate a technical deep-dive PDF for the local CNN emotion recognition project."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether, BaseDocTemplate,
    Frame, PageTemplate
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ─── Palette ───────────────────────────────────────────────────────────────────
DARK     = colors.HexColor("#0f172a")
TEAL     = colors.HexColor("#0d9488")
TEAL2    = colors.HexColor("#0f766e")
SLATE    = colors.HexColor("#334155")
MUTED    = colors.HexColor("#64748b")
LIGHT_BG = colors.HexColor("#f0fdfa")
CODE_BG  = colors.HexColor("#1e293b")
CODE_FG  = colors.HexColor("#e2e8f0")
BORDER   = colors.HexColor("#e2e8f0")
WHITE    = colors.white
AMBER    = colors.HexColor("#d97706")
AMBER_BG = colors.HexColor("#fffbeb")
GREEN    = colors.HexColor("#059669")
GREEN_BG = colors.HexColor("#ecfdf5")
PURPLE   = colors.HexColor("#7c3aed")
PURPLE_BG= colors.HexColor("#f5f3ff")

W, H = A4

# ─── Styles ────────────────────────────────────────────────────────────────────
def S(name, **kw):
    return ParagraphStyle(name, **kw)

cover_title = S("CT", fontName="Helvetica-Bold", fontSize=36, leading=44,
    textColor=WHITE, alignment=TA_CENTER, spaceAfter=10)
cover_sub = S("CS", fontName="Helvetica", fontSize=14, leading=20,
    textColor=colors.HexColor("#99f6e4"), alignment=TA_CENTER, spaceAfter=6)
cover_tag = S("CTG", fontName="Helvetica-Oblique", fontSize=11, leading=16,
    textColor=colors.HexColor("#5eead4"), alignment=TA_CENTER)

h1 = S("H1", fontName="Helvetica-Bold", fontSize=21, leading=27,
    textColor=TEAL2, spaceBefore=16, spaceAfter=8)
h2 = S("H2", fontName="Helvetica-Bold", fontSize=14, leading=19,
    textColor=DARK, spaceBefore=12, spaceAfter=5)
h3 = S("H3", fontName="Helvetica-BoldOblique", fontSize=11.5, leading=16,
    textColor=TEAL, spaceBefore=8, spaceAfter=4)
body = S("Body", fontName="Helvetica", fontSize=10.5, leading=16,
    textColor=colors.HexColor("#1e293b"), spaceAfter=6, alignment=TA_JUSTIFY)
body_sm = S("BSm", fontName="Helvetica", fontSize=9.5, leading=14,
    textColor=SLATE, spaceAfter=4)
mono = S("Mono", fontName="Courier", fontSize=9, leading=13,
    textColor=CODE_FG, backColor=CODE_BG,
    leftIndent=10, rightIndent=10, spaceBefore=3, spaceAfter=3, borderPad=6)
caption = S("Cap", fontName="Helvetica-Oblique", fontSize=9, leading=12,
    textColor=MUTED, alignment=TA_CENTER, spaceAfter=6)
callout = S("Call", fontName="Helvetica", fontSize=10, leading=15,
    textColor=colors.HexColor("#134e4a"), backColor=LIGHT_BG,
    leftIndent=12, rightIndent=12, borderPad=8, spaceBefore=5, spaceAfter=5)
insight = S("Ins", fontName="Helvetica-Bold", fontSize=10, leading=15,
    textColor=colors.HexColor("#3b0764"), backColor=PURPLE_BG,
    leftIndent=12, rightIndent=12, borderPad=8, spaceBefore=5, spaceAfter=5)
note = S("Note", fontName="Helvetica-Oblique", fontSize=10, leading=15,
    textColor=colors.HexColor("#78350f"), backColor=AMBER_BG,
    leftIndent=12, rightIndent=12, borderPad=8, spaceBefore=5, spaceAfter=5)
finding = S("Find", fontName="Helvetica", fontSize=10, leading=15,
    textColor=colors.HexColor("#064e3b"), backColor=GREEN_BG,
    leftIndent=12, rightIndent=12, borderPad=8, spaceBefore=5, spaceAfter=5)

# ─── Helpers ───────────────────────────────────────────────────────────────────
def hr(c=BORDER, t=0.75):
    return HRFlowable(width="100%", thickness=t, color=c, spaceAfter=6, spaceBefore=6)

def sp(n=6):
    return Spacer(1, n)

def P(text, style=body):
    return Paragraph(text, style)

def code(*lines):
    return Paragraph("<br/>".join(lines), mono)

def section(num, title):
    return KeepTogether([hr(TEAL, 2), P(f"{num}. {title}", h1), sp(2)])

def tbl(rows, widths=None, header=True):
    style = [
        ("FONTNAME",       (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",       (0,0), (-1,-1), 9.5),
        ("LEADING",        (0,0), (-1,-1), 14),
        ("VALIGN",         (0,0), (-1,-1), "TOP"),
        ("GRID",           (0,0), (-1,-1), 0.5, BORDER),
        ("LEFTPADDING",    (0,0), (-1,-1), 8),
        ("RIGHTPADDING",   (0,0), (-1,-1), 8),
        ("TOPPADDING",     (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 6),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [WHITE, colors.HexColor("#f8fffd")]),
    ]
    if header:
        style += [
            ("BACKGROUND", (0,0), (-1,0), TEAL),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("TEXTCOLOR",  (0,0), (-1,0), WHITE),
        ]
    cell_rows = [[P(str(c), body_sm) for c in row] for row in rows]
    t = Table(cell_rows, colWidths=widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle(style))
    return t

def layer_box(name, detail, activation, output):
    data = [
        [P(f"<b>{name}</b>", body_sm), P(detail, body_sm),
         P(activation, body_sm), P(output, body_sm)]
    ]
    t = Table(data, colWidths=[3.2*cm, 5*cm, 2.5*cm, 3*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(0,0), TEAL),
        ("TEXTCOLOR",    (0,0),(0,0), WHITE),
        ("FONTNAME",     (0,0),(0,0), "Helvetica-Bold"),
        ("GRID",         (0,0),(-1,-1), 0.5, BORDER),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("TOPPADDING",   (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ]))
    return t

# ─── Page templates ────────────────────────────────────────────────────────────
def cover_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(TEAL2)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#0f766e"))
    canvas.circle(W - 30, H - 30, 130, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#115e59"))
    canvas.circle(50, 90, 90, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#134e4a"))
    canvas.rect(0, 0, W, 70, fill=1, stroke=0)
    canvas.restoreState()

def normal_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(TEAL)
    canvas.rect(0, H - 7, W, 7, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2*cm, 1.2*cm, "Emotion Recognition — Technical Deep-Dive")
    canvas.drawRightString(W - 2*cm, 1.2*cm, f"Page {doc.page}")
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, 1.6*cm, W - 2*cm, 1.6*cm)
    canvas.restoreState()

# ─── Build doc ─────────────────────────────────────────────────────────────────
out = "/Users/leul/Documents/PERSONAL/emotion-recognition/CNN_EmotionRecognition_DeepDive.pdf"

doc = BaseDocTemplate(out, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2.2*cm, bottomMargin=2.2*cm)

cover_frame = Frame(0, 0, W, H, leftPadding=2.5*cm, rightPadding=2.5*cm,
                    topPadding=0, bottomPadding=0, id="cover")
body_frame  = Frame(2*cm, 2*cm, W - 4*cm, H - 4*cm, id="body")

doc.addPageTemplates([
    PageTemplate(id="Cover",  frames=[cover_frame], onPage=cover_bg),
    PageTemplate(id="Normal", frames=[body_frame],  onPage=normal_page),
])

story = []

# ══════════════════════════════════════════════════════════════════════════════
# COVER
# ══════════════════════════════════════════════════════════════════════════════
story.append(sp(150))
story.append(P("Emotion Recognition\nfrom Facial Expressions", cover_title))
story.append(sp(6))
story.append(P("Technical Deep-Dive: Architecture, Training & Cognitive Science", cover_sub))
story.append(sp(28))
story.append(hr(colors.HexColor("#2dd4bf"), 1))
story.append(sp(10))
story.append(P("How the CNN works · How it was trained · Three ML approaches compared", cover_tag))
story.append(sp(8))
story.append(P("Grounded in Ekman, Russell, and connectionist theories of perception", cover_tag))
story.append(sp(20))
story.append(P("Personal Project — Cognitive Science", cover_tag))

story.append(PageBreak())
from reportlab.platypus import NextPageTemplate
story.append(NextPageTemplate("Normal"))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS
# ══════════════════════════════════════════════════════════════════════════════
story.append(P("Table of Contents", h1))
story.append(hr())
toc = [
    ("1", "Project Overview",                               "3"),
    ("2", "The Dataset: FER-2013",                          "3"),
    ("3", "How the CNN Works — Architecture Deep-Dive",     "4"),
    ("4", "How It Was Trained",                             "6"),
    ("5", "Real-Time Detection: How detect.py Works",       "7"),
    ("6", "Three ML Approaches Compared",                   "8"),
    ("7", "Cognitive Science Connections",                  "10"),
    ("8", "Results & What the Confusion Matrix Tells Us",   "12"),
    ("9", "Key Findings & Limitations",                     "13"),
]
toc_rows = [[P(f"<b>{n}.</b>  {t}", body), P(p, body)] for n, t, p in toc]
toc_tbl = Table(toc_rows, colWidths=[W - 4*cm - 30, 30])
toc_tbl.setStyle(TableStyle([
    ("VALIGN",        (0,0),(-1,-1),"MIDDLE"),
    ("LEFTPADDING",   (0,0),(-1,-1),4),
    ("RIGHTPADDING",  (0,0),(-1,-1),4),
    ("TOPPADDING",    (0,0),(-1,-1),5),
    ("BOTTOMPADDING", (0,0),(-1,-1),5),
    ("LINEBELOW",     (0,0),(-1,-1),0.5,BORDER),
    ("ALIGN",         (1,0),(1,-1),"RIGHT"),
    ("ROWBACKGROUNDS",(0,0),(-1,-1),[WHITE,LIGHT_BG]),
]))
story.append(toc_tbl)
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
story.append(section("1", "Project Overview"))
story.append(P(
    "This project is a <b>from-scratch emotion recognition system</b> — a personal exploration "
    "built before the full-stack course demo. It answers a focused question: can a simple "
    "convolutional neural network, trained entirely from random weights on a noisy dataset, "
    "learn to classify human facial expressions?", body))
story.append(P(
    "The answer is yes — imperfectly. And that imperfection is itself instructive, both "
    "technically and cognitively.", body))
story.append(sp(6))

overview = [
    ["Component", "Technology", "Role"],
    ["CNN model",       "TensorFlow / Keras (Sequential)", "Classifies 7 emotions from a 48×48 face patch"],
    ["Face detector",   "OpenCV Haar Cascade", "Locates faces in live webcam frames"],
    ["Training script", "train.py / Kaggle notebook", "Trains the CNN on FER-2013 for 50 epochs"],
    ["Inference script","detect.py", "Real-time webcam loop with emotion overlay"],
    ["Model file",      "emotion_model.h5", "Saved Keras weights (~10 MB)"],
    ["Comparison",      "comparison_notebook.ipynb", "HOG+SVM vs CNN vs Transfer Learning"],
]
story.append(tbl(overview, widths=[3.5*cm, 5.5*cm, W - 4*cm - 9*cm]))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ══════════════════════════════════════════════════════════════════════════════
story.append(section("2", "The Dataset: FER-2013"))
story.append(P(
    "The model was trained exclusively on <b>FER-2013</b> (Facial Expression Recognition 2013), "
    "a dataset published by Goodfellow et al. at a 2013 ICML competition. It is the foundational "
    "benchmark for facial emotion recognition.", body))
story.append(sp(6))

fer_props = [
    ["Property", "Detail"],
    ["Total images",      "35,887"],
    ["Image format",      "Grayscale, 48×48 pixels"],
    ["Storage format",    "CSV — each row is a pixel string (2,304 space-separated integers, 0–255)"],
    ["Classes",           "7: Angry (0), Disgust (1), Fear (2), Happy (3), Sad (4), Surprise (5), Neutral (6)"],
    ["Dataset splits",    "Training (~28,709) / PublicTest (~3,589) / PrivateTest (~3,589)"],
    ["Label method",      "Single annotator per image — one person decided the emotion label"],
    ["Estimated noise",   "~30% mislabel rate — a known and widely reported flaw"],
    ["Class balance",     "Heavily skewed: Happy (~25%), Neutral (~25%), Sad/Anger (~12% each), Disgust (~4%)"],
    ["Origin",            "Google image search results, automatically gathered and manually labelled"],
]
story.append(tbl(fer_props, widths=[4.5*cm, W - 4*cm - 4.5*cm]))
story.append(sp(8))

story.append(P("2.1 What the Pixel Format Looks Like", h2))
story.append(P("Each image is stored as a single string of 2,304 integers in the CSV:", body))
story.append(code(
    "# Row in fer2013.csv:",
    "emotion,pixels,Usage",
    "0,70 80 82 72 58 58 60 63 54 58 60 48 89 115 121 ...,Training",
    "",
    "# To reconstruct the image in Python:",
    "import numpy as np",
    "pixels = np.array('70 80 82 ...'.split(), dtype=np.float32)",
    "image  = pixels.reshape(48, 48)       # 2D grayscale array",
    "image  = image / 255.0                # normalise to [0, 1]",
    "image  = image.reshape(48, 48, 1)     # add channel dim for Keras",
))
story.append(sp(8))

story.append(P("2.2 Why the Noise Matters", h2))
story.append(P(
    "The ~30% mislabel rate is not a minor footnote — it sets a <b>ceiling on accuracy</b>. "
    "If 30% of training labels are wrong, the model is learning from contradictory signals. "
    "It cannot outperform its training data's signal quality, no matter how deep the architecture. "
    "This is the central empirical finding of this project line of work: label quality gates "
    "model performance more than architecture does.", body))
story.append(P(
    "Cognitively, this mirrors how humans develop emotion recognition: a child given inconsistent "
    "feedback ('that face means angry' for what adults would call fear) will develop a miscalibrated "
    "emotion detector — regardless of how many examples they see.", note))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 3. CNN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
story.append(section("3", "How the CNN Works — Architecture Deep-Dive"))
story.append(P(
    "The model is a <b>Sequential Convolutional Neural Network</b> built in Keras. It takes a "
    "48×48 grayscale image as input and outputs a probability distribution over 7 emotion classes. "
    "Every weight in the network is initialised randomly and learned entirely from FER-2013 — "
    "there is no transfer learning, no pretrained backbone.", body))
story.append(sp(8))

story.append(P("3.1 Layer-by-Layer Breakdown", h2))
story.append(P(
    "The network has three convolutional blocks followed by a fully-connected classifier head. "
    "Here is what each layer does and why it is there:", body))
story.append(sp(6))

# Conv block headers
story.append(P("<b>Block 1 — Low-Level Feature Detection</b>", h3))
for row in [
    ("Conv2D(32, 3×3)\nrelu", "32 filters, each 3×3 pixels. Learns to detect low-level features: edges, "
     "colour gradients, simple textures. Each filter slides over the entire image producing a 46×46 feature map.", "ReLU", "32 × 46 × 46"),
    ("BatchNormalization", "Normalises each feature map's activations to mean=0, std=1. "
     "Reduces internal covariate shift — stabilises and speeds up training.", "—", "32 × 46 × 46"),
    ("MaxPooling2D(2×2)", "Takes the maximum value in each 2×2 patch. Halves spatial dimensions. "
     "Provides translation invariance — a feature detected slightly off-centre still fires.", "—", "32 × 23 × 23"),
    ("Dropout(0.25)", "During training, randomly zeros 25% of units. "
     "Forces the network to learn redundant representations. Reduces overfitting.", "—", "32 × 23 × 23"),
]:
    story.append(layer_box(*row))
    story.append(sp(2))

story.append(sp(6))
story.append(P("<b>Block 2 — Mid-Level Feature Composition</b>", h3))
for row in [
    ("Conv2D(64, 3×3)\nrelu", "64 filters. Each filter now operates on the 32 feature maps from Block 1, "
     "so it learns combinations of low-level features: curves, corners, partial facial structures.", "ReLU", "64 × 21 × 21"),
    ("BatchNormalization", "Same purpose as Block 1.", "—", "64 × 21 × 21"),
    ("MaxPooling2D(2×2)", "Halves again.", "—", "64 × 10 × 10"),
    ("Dropout(0.25)", "25% dropout.", "—", "64 × 10 × 10"),
]:
    story.append(layer_box(*row))
    story.append(sp(2))

story.append(sp(6))
story.append(P("<b>Block 3 — High-Level Facial Feature Detection</b>", h3))
for row in [
    ("Conv2D(128, 3×3)\nrelu", "128 filters. Now composing mid-level features into high-level representations: "
     "eye shapes, mouth curvature, brow position — abstract descriptors relevant to emotion.", "ReLU", "128 × 8 × 8"),
    ("BatchNormalization", "Same purpose.", "—", "128 × 8 × 8"),
    ("MaxPooling2D(2×2)", "Halves again.", "—", "128 × 4 × 4"),
    ("Dropout(0.40)", "Higher 40% dropout here because the layer is most prone to overfitting "
     "— 128 filters × 8×8 receptive fields = many parameters.", "—", "128 × 4 × 4"),
]:
    story.append(layer_box(*row))
    story.append(sp(2))

story.append(sp(6))
story.append(P("<b>Classifier Head</b>", h3))
for row in [
    ("Flatten", "Unrolls the 128×4×4 tensor into a 2,048-element vector. "
     "Converts spatial feature maps into a format the dense layers can process.", "—", "2,048"),
    ("Dense(256)\nrelu", "Fully connected layer. Learns non-linear combinations of the "
     "2,048 features that are predictive of emotion categories.", "ReLU", "256"),
    ("Dropout(0.50)", "Aggressive 50% dropout on the dense layer — "
     "this layer has the most parameters and is most prone to memorising training data.", "—", "256"),
    ("Dense(7)\nsoftmax", "Output layer. 7 neurons, one per emotion. Softmax ensures outputs "
     "are probabilities summing to 1.0. The argmax of this vector is the predicted emotion.", "Softmax", "7"),
]:
    story.append(layer_box(*row))
    story.append(sp(2))

story.append(sp(8))
story.append(P("3.2 Architecture Summary", h2))
arch_data = [
    ["Stage", "Layer", "Output shape", "Parameters"],
    ["Input",         "—",                    "48 × 48 × 1",   "0"],
    ["Block 1",       "Conv2D(32) + BN + Pool + Drop", "23 × 23 × 32",  "~9,500"],
    ["Block 2",       "Conv2D(64) + BN + Pool + Drop", "10 × 10 × 64",  "~37,000"],
    ["Block 3",       "Conv2D(128) + BN + Pool + Drop","4 × 4 × 128",   "~148,000"],
    ["Classifier",    "Flatten → Dense(256) → Dense(7)","7",            "~525,000"],
    ["<b>Total</b>",  "—",                    "—",             "<b>~720,000</b>"],
]
story.append(tbl(arch_data, widths=[2.5*cm, 6*cm, 3.5*cm, W - 4*cm - 12*cm]))
story.append(sp(6))
story.append(P(
    "~720k parameters is small by modern standards — EfficientNet-B0 has 5.3M. This is intentional: "
    "the model is designed to train quickly on CPU/Kaggle without GPU for hours.", callout))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ══════════════════════════════════════════════════════════════════════════════
story.append(section("4", "How It Was Trained"))

story.append(P("4.1 Training Configuration", h2))
train_cfg = [
    ["Hyperparameter", "Value", "Reason"],
    ["Optimizer",    "Adam",                   "Adaptive learning rates — works well without manual LR tuning"],
    ["Loss function","Categorical cross-entropy","Standard multi-class classification loss"],
    ["Metric",       "Accuracy",               "Simple fraction of correctly classified samples"],
    ["Epochs",       "50",                     "Enough for convergence on this dataset size"],
    ["Batch size",   "64",                     "Balance between training stability and memory usage"],
    ["Val split",    "20% (random_state=42)",  "Held-out for monitoring overfitting during training"],
    ["Weight init",  "Glorot uniform (default)","Xavier initialisation — keeps gradients stable at start"],
    ["Input range",  "[0, 1] (÷255)",          "Normalised from [0, 255] — required for stable gradient flow"],
]
story.append(tbl(train_cfg, widths=[3.5*cm, 3.5*cm, W - 4*cm - 7*cm]))
story.append(sp(8))

story.append(P("4.2 What Happens During Training", h2))
story.append(P(
    "Each training epoch processes all ~28,000 training images in batches of 64. For each batch:", body))
story.append(code(
    "Forward pass:   image → Conv blocks → Dense → softmax → predicted probabilities",
    "Loss computed:  cross_entropy(predicted_probs, true_one_hot_label)",
    "Backward pass:  compute gradient of loss w.r.t. every weight (backpropagation)",
    "Weight update:  Adam adjusts each weight by its gradient × adaptive learning rate",
    "Repeat:         next batch, then next epoch",
))
story.append(sp(8))

story.append(P("4.3 Regularisation Strategy", h2))
story.append(P(
    "Three techniques prevent the network from memorising the training data:", body))
reg_data = [
    ["Technique", "Where applied", "What it does"],
    ["Batch Normalization",
     "After each Conv2D",
     "Stabilises activations, acts as mild regulariser, allows higher learning rates"],
    ["Dropout(0.25 / 0.40)",
     "After each pooling layer",
     "Randomly disables neurons during training, preventing co-adaptation"],
    ["Dropout(0.50)",
     "After Dense(256)",
     "Heaviest regularisation on the most parameter-rich layer"],
]
story.append(tbl(reg_data, widths=[3.5*cm, 4*cm, W - 4*cm - 7.5*cm]))
story.append(sp(6))
story.append(P(
    "Note: Dropout is only active during training. At inference time (detect.py, evaluation), "
    "all neurons are active and their outputs are scaled by (1 - dropout_rate) to compensate.", note))
story.append(sp(8))

story.append(P("4.4 Training Platform", h2))
story.append(P(
    "Two notebooks exist for training:", body))
platform_data = [
    ["Notebook", "Platform", "Use case"],
    ["kaggle_notebook.ipynb",    "Kaggle (free GPU — Tesla P100)", "Initial training with full dataset"],
    ["comparison_notebook.ipynb","Kaggle",                         "Side-by-side comparison of 3 ML approaches"],
    ["train.py",                 "Local (CPU)",                    "Alternate local training script"],
]
story.append(tbl(platform_data, widths=[4.5*cm, 4*cm, W - 4*cm - 8.5*cm]))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 5. REAL-TIME DETECTION
# ══════════════════════════════════════════════════════════════════════════════
story.append(section("5", "Real-Time Detection: How detect.py Works"))
story.append(P(
    "Once trained, the model is used in <b>detect.py</b> — a real-time webcam loop that draws "
    "emotion labels and probability bars directly onto the video feed using OpenCV.", body))
story.append(sp(6))

story.append(P("5.1 The Detection Pipeline", h2))
pipeline = [
    ("Read frame",       "cap.read()",
     "OpenCV captures a BGR frame from webcam (default device 0)"),
    ("Grayscale convert","cv2.cvtColor(frame, BGR2GRAY)",
     "Convert to single-channel grayscale — the model was trained on grayscale images"),
    ("Face detection",   "face_cascade.detectMultiScale(gray, ...)",
     "Haar Cascade scans for frontal faces. Returns bounding boxes (x, y, w, h)"),
    ("ROI crop & resize","roi = gray[y:y+h, x:x+w]; cv2.resize(roi, (48,48))",
     "Crop the face region, resize to exactly 48×48 to match training input size"),
    ("Normalise",        "roi = roi.astype(np.float32) / 255.0",
     "Scale pixels to [0,1] — same normalisation as during training"),
    ("Reshape for Keras","roi.reshape(1, 48, 48, 1)",
     "Add batch dimension (1) and channel dimension (1) — Keras expects 4D tensors"),
    ("Predict",          "model.predict(roi, verbose=0)[0]",
     "Returns array of 7 probabilities. Takes ~5–15ms on CPU."),
    ("Draw overlay",     "cv2.rectangle + cv2.putText",
     "Draws coloured bounding box, emotion label, confidence %, and probability bars"),
    ("Display",          "cv2.imshow(...)",
     "Shows the annotated frame in a desktop window. Press Q to quit."),
]
pipe_data = [["Step", "Code", "What happens"]] + [[P(f"<b>{s}</b>",body_sm), P(c,mono), P(d,body_sm)] for s,c,d in pipeline]
pipe_tbl = Table([[P(f"<b>{row[0]}</b>",body_sm), P(str(row[1]),body_sm if i>0 else body_sm), P(str(row[2]),body_sm)] for i,row in enumerate(pipeline)],
    colWidths=[2.5*cm, 5*cm, W - 4*cm - 7.5*cm])
pipe_tbl.setStyle(TableStyle([
    ("FONTNAME",      (0,0),(-1,-1),"Helvetica"),
    ("FONTSIZE",      (0,0),(-1,-1),9),
    ("LEADING",       (0,0),(-1,-1),13),
    ("VALIGN",        (0,0),(-1,-1),"TOP"),
    ("GRID",          (0,0),(-1,-1),0.5,BORDER),
    ("LEFTPADDING",   (0,0),(-1,-1),6),
    ("RIGHTPADDING",  (0,0),(-1,-1),6),
    ("TOPPADDING",    (0,0),(-1,-1),5),
    ("BOTTOMPADDING", (0,0),(-1,-1),5),
    ("ROWBACKGROUNDS",(0,0),(-1,-1),[WHITE,LIGHT_BG]),
    ("BACKGROUND",    (0,0),(0,-1),colors.HexColor("#f0fdfa")),
    ("FONTNAME",      (0,0),(0,-1),"Helvetica-Bold"),
]))
story.append(pipe_tbl)
story.append(sp(8))

story.append(P("5.2 Haar Cascade vs RetinaFace — Why the Detector Matters", h2))
detector_data = [
    ["", "Haar Cascade (this project)", "RetinaFace (full-stack project)"],
    ["Type",        "Classical computer vision (2001)", "Deep learning (2020)"],
    ["Speed",       "Very fast (~1ms per frame)",       "Slower (~50–200ms)"],
    ["Accuracy",    "Low — misses off-angle, occluded, non-frontal faces", "High — detects faces at many angles"],
    ["Crop quality","Loose bounding box — includes chin, forehead, background", "Tight crop with 5 landmark alignment"],
    ["False positives","Common — can trigger on shadows and patterns", "Rare"],
    ["Implication", "Poor crop quality feeds noise into the emotion model", "Clean aligned crops improve emotion accuracy"],
]
story.append(tbl(detector_data, widths=[2.5*cm, 5.5*cm, W - 4*cm - 8*cm]))
story.append(sp(6))
story.append(P(
    "The detector is a significant bottleneck. Even if the emotion CNN achieves 60% on clean "
    "crops, a loose or slightly misaligned Haar crop can push real-world accuracy much lower. "
    "This was a primary motivation for upgrading to RetinaFace in the full-stack project.", callout))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 6. THREE APPROACHES COMPARED
# ══════════════════════════════════════════════════════════════════════════════
story.append(section("6", "Three ML Approaches Compared"))
story.append(P(
    "The <b>comparison_notebook.ipynb</b> goes beyond the CNN and evaluates three fundamentally "
    "different machine learning paradigms on the same dataset. This comparison is the most "
    "cognitively instructive part of the project.", body))
story.append(sp(8))

# HOG + SVM
story.append(P("6.1 Approach 1 — HOG + SVM (Hand-Crafted Features)", h2))
story.append(P(
    "<b>HOG (Histogram of Oriented Gradients)</b> is a classical computer vision feature extractor. "
    "It does not learn — it applies a fixed mathematical formula to every image:", body))
story.append(code(
    "For each image:",
    "  1. Divide the 48×48 image into 8×8 pixel cells",
    "  2. In each cell, compute a histogram of gradient orientations (9 bins)",
    "  3. Normalise across 2×2 blocks of cells",
    "  4. Concatenate all histograms → 1 feature vector per image",
    "",
    "Result: each face becomes a fixed-length vector capturing local edge directions",
    "HOG feature vector size: ~1,764 dimensions",
))
story.append(P(
    "An <b>SVM (Support Vector Machine)</b> with RBF kernel then learns to draw a decision "
    "boundary in this 1,764-dimensional feature space, separating the 7 emotion classes.", body))
hog_data = [
    ["Property", "Detail"],
    ["Features",    "Hand-crafted HOG descriptors — edge orientations, not learned"],
    ["Classifier",  "SVM with RBF kernel, C=10, gamma='scale'"],
    ["Training",    "Feature extraction (minutes) + SVM fit (minutes on CPU)"],
    ["Approximate accuracy", "~35–42% on FER-2013 validation"],
    ["Strength",    "Interpretable, fast at inference, no GPU needed"],
    ["Weakness",    "Cannot capture non-linear spatial relationships, misses facial dynamics"],
]
story.append(tbl(hog_data, widths=[3.5*cm, W - 4*cm - 3.5*cm]))
story.append(sp(10))

# Custom CNN
story.append(P("6.2 Approach 2 — Custom CNN (Learned Features)", h2))
story.append(P(
    "This is the same architecture described in Section 3 — the model in <b>emotion_model.h5</b>. "
    "Unlike HOG, the CNN <b>learns which features matter</b> from the data itself through "
    "backpropagation.", body))
cnn_data = [
    ["Property", "Detail"],
    ["Features",    "Learned by 32/64/128 convolutional filters from random initialisation"],
    ["Architecture","Conv2D → BN → MaxPool → Dropout (×3) → Dense(256) → Dense(7)"],
    ["Training",    "50 epochs, Adam optimizer, batch size 64"],
    ["Approximate accuracy", "~55–62% on FER-2013 validation"],
    ["Strength",    "Learns emotion-relevant features automatically, better than HOG at non-linear patterns"],
    ["Weakness",    "No pretrained knowledge, limited by FER-2013 label noise"],
]
story.append(tbl(cnn_data, widths=[3.5*cm, W - 4*cm - 3.5*cm]))
story.append(sp(10))

# Transfer Learning
story.append(P("6.3 Approach 3 — Transfer Learning (MobileNetV2)", h2))
story.append(P(
    "<b>MobileNetV2</b>, pretrained on ImageNet (1.2 million images, 1,000 classes), is used as "
    "a frozen feature extractor. Only the new classification head is trained on FER-2013:", body))
story.append(code(
    "# Architecture in comparison_notebook.ipynb:",
    "base_model = MobileNetV2(input_shape=(48,48,3), include_top=False, weights='imagenet')",
    "base_model.trainable = False   # freeze all 2.2M pretrained weights",
    "",
    "model = Sequential([",
    "    base_model,                # pretrained feature extractor",
    "    GlobalAveragePooling2D(),  # collapse spatial dims",
    "    Dense(128, activation='relu'),",
    "    Dropout(0.4),",
    "    Dense(7, activation='softmax'),",
    "])",
    "",
    "# Input: grayscale → RGB by repeating channel (np.repeat(..., 3, axis=-1))",
))
tl_data = [
    ["Property", "Detail"],
    ["Backbone",    "MobileNetV2 — lightweight depthwise-separable CNN from Google (2018)"],
    ["Pretrained on","ImageNet (1.2M images, 1000 classes)"],
    ["Trainable params","Head only: ~16,000 (backbone frozen at 2.2M)"],
    ["Training",    "30 epochs, Adam, batch 64, frozen backbone"],
    ["Approximate accuracy", "~58–65% on FER-2013 validation"],
    ["Strength",    "Learns fast, ImageNet features transfer to faces"],
    ["Weakness",    "MobileNetV2 expects 96×96+ input — 48×48 is below optimal resolution"],
]
story.append(tbl(tl_data, widths=[3.5*cm, W - 4*cm - 3.5*cm]))
story.append(sp(8))

summary_data = [
    ["Approach",             "Features",  "Trainable Params", "Approx. Accuracy", "Training Time"],
    ["HOG + SVM",            "Fixed (math formula)",    "~SVM support vectors", "~35–42%", "~5 min (CPU)"],
    ["Custom CNN",           "Learned from scratch",    "~720,000",             "~55–62%", "~30 min (GPU)"],
    ["Transfer (MobileNetV2)","Pretrained ImageNet",    "~16,000 (head only)",  "~58–65%", "~10 min (GPU)"],
]
story.append(tbl(summary_data, widths=[3.5*cm, 3.5*cm, 3.5*cm, 3*cm, W - 4*cm - 13.5*cm]))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 7. COGNITIVE SCIENCE
# ══════════════════════════════════════════════════════════════════════════════
story.append(section("7", "Cognitive Science Connections"))

story.append(P("7.1 Ekman's Basic Emotions Theory — The Foundation", h2))
story.append(P(
    "Every design decision in this project — the 7 emotion categories, the dataset choice, "
    "the evaluation metrics — is grounded in <b>Paul Ekman's Basic Emotions Theory (1971)</b>.", body))
story.append(P(
    "Ekman argued, based on cross-cultural studies with isolated Papua New Guinean communities, "
    "that six emotions are <b>universal and biologically innate</b> — not learned culturally. "
    "Each is associated with a specific, reproducible configuration of facial muscles (Action "
    "Units in his Facial Action Coding System, FACS):", body))

ekman = [
    ["Emotion",   "Key facial features",                              "FACS Action Units (approx.)"],
    ["Happiness", "Raised lip corners, raised cheeks, crow's feet",   "AU6 + AU12"],
    ["Sadness",   "Inner brow raise, lip corner depression",          "AU1 + AU4 + AU15"],
    ["Fear",      "Raised brows, wide eyes, horizontal lip stretch",  "AU1 + AU2 + AU4 + AU5 + AU20"],
    ["Anger",     "Brow lowering, raised upper lid, pressed lips",    "AU4 + AU5 + AU7 + AU23"],
    ["Disgust",   "Nose wrinkle, upper lip raise, lip corner depression","AU9 + AU15 + AU16"],
    ["Surprise",  "Raised brows, wide eyes, dropped jaw",             "AU1 + AU2 + AU5 + AU26"],
    ["Neutral",   "No significant muscle activation",                 "—"],
]
story.append(tbl(ekman, widths=[2.5*cm, 6.5*cm, W - 4*cm - 9*cm]))
story.append(sp(6))
story.append(P(
    "The CNN does not know about Action Units — but its learned filters in the third conv block "
    "likely correspond functionally to detecting local facial regions (brow, mouth corners, eye "
    "width) that overlap with these FACS descriptors.", callout))
story.append(sp(8))

story.append(P("7.2 The Three ML Approaches as Theories of Cognition", h2))
story.append(P(
    "The comparison notebook explicitly frames the three ML approaches as analogies for three "
    "competing theories in cognitive science:", body))
story.append(sp(4))

cog_analogies = [
    ["ML Approach", "Cognitive Theory", "Analogy", "Implication"],
    ["HOG + SVM",
     "Rule-based / Symbolic AI\n(Newell & Simon, 1976)",
     "Early cognitive models assumed fixed feature detectors — like Gabor filters tuned to "
     "specific edge orientations. Emotion recognition = detecting specific geometric configurations.",
     "Fails on ambiguous or stylised expressions. Rules cannot capture the full variance of "
     "natural facial expressions."],
    ["Custom CNN",
     "Connectionism /\nNeural network theory\n(Rumelhart et al., 1986)",
     "The CNN's layered hierarchy mirrors the <b>visual cortex</b>: V1 detects edges (Conv1), "
     "V2 combines into curves (Conv2), IT cortex integrates into object/face recognition (Conv3). "
     "Features are not hand-designed — they emerge from experience (training data).",
     "Performance is bounded by the quality and quantity of training experience. Matches "
     "developmental theories: the visual system is shaped by what it is exposed to."],
    ["Transfer Learning",
     "Schema theory /\nExperience-dependent\nplasticity",
     "MobileNetV2 pre-trained on ImageNet represents prior visual experience. Fine-tuning "
     "represents applying existing schemas to a new task. Analogous to the <b>fusiform face "
     "area</b> — the brain's face-selective region, which is active from birth and refined "
     "through experience.",
     "Prior knowledge accelerates new learning. This is consistent with cognitive load theory "
     "and schema-based learning: experts learn new variants of familiar patterns faster."],
]
story.append(tbl(cog_analogies, widths=[2.5*cm, 3*cm, 5.5*cm, W - 4*cm - 11*cm]))
story.append(sp(8))

story.append(P("7.3 Russell's Circumplex Model and the Probability Outputs", h2))
story.append(P(
    "While the CNN outputs discrete labels, the <b>probability distribution</b> over all 7 emotions "
    "is implicitly dimensional. Russell's (1980) circumplex model places emotions on two axes:", body))
story.append(code(
    "Valence axis:  negative ← [Fear, Anger, Disgust, Sadness] ← neutral → [Happy] → positive",
    "Arousal axis:  low     ← [Neutral, Sadness] ← neutral → [Surprise, Fear, Anger] → high",
    "",
    "Example inference output:",
    "  happy: 72%, surprise: 18%, neutral: 6%, fear: 2%, sad: 1%, angry: 1%",
    "  → High valence (happy dominates), high arousal (surprise component)",
    "  → Maps to top-right of Russell's circumplex (excited/elated region)",
))
story.append(sp(6))
story.append(P(
    "This means the emotion probabilities are not just a classification tool — they are a "
    "rough probabilistic coordinate in a 2D affective space. The model is implicitly doing "
    "dimensional affect estimation even though it was only trained on categorical labels.", insight))
story.append(sp(8))

story.append(P("7.4 The Memory–Emotion Link", h2))
story.append(P(
    "The comparison notebook explicitly connects model performance patterns to the memory "
    "literature. The pattern observed across all three ML approaches:", body))
mem_data = [
    ["Observation", "Cognitive science explanation"],
    ["Happy and Surprise\nconsistently highest accuracy",
     "Emotionally salient stimuli are encoded more strongly in memory (McGaugh, 2000: "
     "amygdala modulation of hippocampal consolidation). Happy faces may also be most "
     "frequently encountered, improving recognition."],
    ["Fear and Disgust\nconsistently confused",
     "Humans also confuse Fear and Disgust under time pressure (Ekman & Friesen, 1978). "
     "Both involve threat-relevant processing, and their facial configurations share "
     "features (brow raising, eye widening)."],
    ["Neutral heavily\nmispredicted",
     "Models trained on imbalanced data default to the majority class. But cognitively, "
     "neutral is also the hardest for humans: it is defined by absence of expression, "
     "meaning subtle variations in resting face anatomy create noise."],
]
story.append(tbl(mem_data, widths=[3.5*cm, W - 4*cm - 3.5*cm]))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 8. RESULTS
# ══════════════════════════════════════════════════════════════════════════════
story.append(section("8", "Results & What the Confusion Matrix Tells Us"))

story.append(P("8.1 Accuracy Summary", h2))
results_data = [
    ["Model / Approach", "Val Accuracy", "Key Observation"],
    ["HOG + SVM",         "~35–42%",  "Cannot handle non-linear facial variation"],
    ["Custom CNN",        "~55–62%",  "Learns better features but bounded by FER-2013 noise"],
    ["Transfer (MobileNetV2)","~58–65%","ImageNet features help, but 48×48 resolution is too small for MobileNetV2"],
    ["HSEmotion (AffectNet)","~63–66%","Same EfficientNet-B0, 10× more data, cleaner labels (from full-stack project)"],
    ["Fine-tuned EfficientNet (FER+)","~82%","Same images as FER-2013, but 10 annotators per image — quality labels win"],
]
story.append(tbl(results_data, widths=[4.5*cm, 2.5*cm, W - 4*cm - 7*cm]))
story.append(sp(8))

story.append(P("8.2 How to Read a Confusion Matrix", h2))
story.append(P(
    "The confusion matrix is a 7×7 grid. Row = true emotion, column = predicted emotion. "
    "The diagonal = correct predictions. Off-diagonal cells = confusions.", body))
story.append(code(
    "        Angry Disgust Fear Happy  Sad  Surprise Neutral",
    "Angry   [150]   8     12    5    20     3        45     ← often confused with Neutral",
    "Disgust [15]   [62]   18    2    10     1        20     ← confused with Angry/Fear",
    "Fear    [12]   10    [95]   3    15    30        28     ← confused with Surprise (brow raise)",
    "Happy   [2]     1     1   [410]   5     8         9     ← highest recall class",
    "Sad     [25]   8     10    3   [160]    2        55     ← confused with Neutral",
    "Surprise[5]    2     28    8     2    [190]       10    ← confused with Fear",
    "Neutral [40]   8     20   10    45     5        [280]   ← confused with Sad/Angry",
))
story.append(sp(4))
story.append(P("(Illustrative values — actual numbers depend on the trained model.)", caption))
story.append(sp(6))

confusion_insights = [
    ("Fear ↔ Surprise confusion",
     "Both involve raised brows and wide eyes (AU1, AU2, AU5). The key distinguishing "
     "feature is mouth: open/dropped jaw = surprise, horizontal stretch = fear. At 48×48 "
     "resolution, this subtle difference is hard to detect."),
    ("Angry ↔ Neutral confusion",
     "Mild anger can look like a neutral resting face, especially in FER-2013 images where "
     "expressions are often subtle or ambiguous. Also reflects FER-2013's neutral bias."),
    ("Disgust low recall",
     "Disgust is the smallest class (~4% of FER-2013). The model sees very few examples, "
     "so it under-learns this category and predicts it as Angry or Fear."),
    ("Happy highest recall",
     "Happy has the most training examples, the clearest visual signature (bilateral lip "
     "raise + cheek raise), and is the most culturally practiced posed expression."),
]
for conf, explanation in confusion_insights:
    row = Table(
        [[P(f"<b>{conf}</b>", body_sm), P(explanation, body_sm)]],
        colWidths=[4*cm, W - 4*cm - 4*cm]
    )
    row.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(0,0), LIGHT_BG),
        ("GRID",         (0,0),(-1,-1), 0.5, BORDER),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("TOPPADDING",   (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ("VALIGN",       (0,0),(-1,-1), "TOP"),
    ]))
    story.append(row)
    story.append(sp(4))

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 9. FINDINGS & LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════════
story.append(section("9", "Key Findings & Limitations"))

story.append(P("9.1 Key Findings", h2))
findings = [
    ("Label quality gates accuracy more than architecture",
     "The Custom CNN (~60%) and Transfer Learning (~65%) both use FER-2013's noisy labels. "
     "The Fine-tuned EfficientNet achieves ~82% on the same images with better labels. "
     "No architectural improvement overcomes a ~30% mislabel rate."),
    ("The HOG–CNN accuracy gap validates deep learning for face perception",
     "HOG at 35–42% vs CNN at 55–62% on the same images confirms that emotion recognition "
     "requires learning non-linear, context-dependent features — not just edge statistics. "
     "This validates connectionist over rule-based theories for this task."),
    ("Transfer learning is architecture-bounded",
     "MobileNetV2 performs only marginally better than the Custom CNN, likely because "
     "48×48 input is far below its optimal 96×96+ resolution. The ImageNet features "
     "need sufficient spatial detail to be useful."),
    ("The Happy–Neutral distribution skew is the dominant source of error",
     "Both classes are over-represented in FER-2013 and the model collapses many "
     "ambiguous expressions into these two categories."),
]
for title, desc in findings:
    story.append(P(f"<b>{title}</b>", body))
    story.append(P(desc, callout))
    story.append(sp(4))

story.append(sp(6))
story.append(P("9.2 Limitations", h2))
limits = [
    ["Limitation", "Impact"],
    ["Static single frame",
     "Real human emotion perception integrates dynamics over time, body language, voice, "
     "and context. A still 48×48 grayscale crop contains almost none of this information."],
    ["Grayscale only",
     "Skin tone, colour changes due to blushing or pallor, and other chromatic cues are "
     "completely discarded. Colour is a meaningful affect signal."],
    ["Haar Cascade detector",
     "Poor detection on off-angle faces, occlusion, poor lighting. The emotion model can "
     "only be as good as the face crop it receives."],
    ["FER-2013 label noise",
     "~30% estimated mislabel rate. The model may have learned incorrect associations "
     "that happen to produce reasonable aggregate accuracy."],
    ["Demographic bias",
     "FER-2013 was collected via Google image search and skews toward certain demographics, "
     "expression styles, and cultural display norms."],
    ["Categorical constraint",
     "Real emotional states are mixed, dimensional, and dynamic. Forcing a face into one of "
     "7 discrete labels discards nuance and introduces artificial certainty."],
]
story.append(tbl(limits, widths=[4*cm, W - 4*cm - 4*cm]))

# Footer
story.append(sp(16))
story.append(hr(TEAL, 2))
story.append(sp(6))
refs_data = [
    [P("References", h3), P("Related Project", h3)],
    [
        P("Ekman, P. (1971). Universals and cultural differences in facial expressions of emotion.<br/>"
          "Russell, J.A. (1980). A circumplex model of affect.<br/>"
          "McGaugh, J.L. (2000). Memory: A century of consolidation.<br/>"
          "Goodfellow et al. (2013). Challenges in representation learning (FER-2013).<br/>"
          "Rumelhart et al. (1986). Learning representations by back-propagating errors.<br/>"
          "Sandler et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.", body_sm),
        P("Full-stack version: <b>Facial-emotion-recognition</b> (Downloads)<br/>"
          "Adds: FastAPI backend, React UI, RetinaFace detector,<br/>"
          "HSEmotion model, EfficientNet-B0 fine-tuned on FER+,<br/>"
          "model comparison UI, 82% validation accuracy.", body_sm),
    ]
]
refs_tbl = Table(refs_data, colWidths=[(W-4*cm)*0.58, (W-4*cm)*0.42])
refs_tbl.setStyle(TableStyle([
    ("VALIGN",      (0,0),(-1,-1),"TOP"),
    ("LEFTPADDING", (0,0),(-1,-1),0),
    ("RIGHTPADDING",(0,0),(-1,-1),8),
    ("TOPPADDING",  (0,0),(-1,-1),0),
    ("BOTTOMPADDING",(0,0),(-1,-1),0),
]))
story.append(refs_tbl)

doc.build(story)
print(f"PDF generated: {out}")
