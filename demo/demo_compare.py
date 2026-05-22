import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from skimage.feature import hog

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
COLORS = {
    'Angry':     (0,   0,   255),
    'Disgust':   (0,   140,   0),
    'Fear':      (128,   0, 128),
    'Happy':     (0,   255, 255),
    'Sad':       (255, 100,   0),
    'Surprise':  (0,   165, 255),
    'Neutral':   (200, 200, 200),
}

def _load_tl_model(path):
    # Rebuild with Functional API to avoid Sequential-submodel deserialization
    # issues across Keras versions, then restore weights from the saved file.
    base = applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights=None)
    inp  = layers.Input(shape=(96, 96, 3))
    x    = base(inp, training=False)
    x    = layers.GlobalAveragePooling2D()(x)
    x    = layers.Dense(128, activation='relu')(x)
    x    = layers.Dropout(0.4)(x)
    out  = layers.Dense(7, activation='softmax')(x)
    model = models.Model(inp, out)
    model.load_weights(path)
    return model

print("Loading models...")
svm_pipeline = joblib.load('hog_svm_model.pkl')
cnn_model    = tf.keras.models.load_model('emotion_model.h5')
tl_model     = _load_tl_model('tl_emotion_model.keras')
print("All models loaded.")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def predict_svm(face_48):
    feat = hog(face_48, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), visualize=False).reshape(1, -1)
    return svm_pipeline.predict_proba(feat)[0]


def predict_cnn(face_48):
    inp = face_48.astype(np.float32) / 255.0
    return cnn_model.predict(inp.reshape(1, 48, 48, 1), verbose=0)[0]


def predict_tl(face_48):
    resized = cv2.resize(face_48, (96, 96))
    inp = np.repeat(resized[..., np.newaxis], 3, axis=-1).astype(np.float32) / 255.0
    return tl_model.predict(inp.reshape(1, 96, 96, 3), verbose=0)[0]


MODELS = [
    ('HOG + SVM',         predict_svm, ( 30, 140, 255)),
    ('Custom CNN',        predict_cnn, ( 40, 200,  80)),
    ('Transfer Learning', predict_tl,  (200, 100, 255)),
]

PANEL_W   = 190
PANEL_GAP = 6
BAR_H     = 14
BAR_GAP   = 4
PAD       = 8


def draw_panel(frame, probs, name, header_color, px, py):
    top_idx = int(np.argmax(probs))
    panel_h = 26 + len(EMOTIONS) * (BAR_H + BAR_GAP) + 4

    # header
    cv2.rectangle(frame, (px, py), (px + PANEL_W, py + 22), header_color, -1)
    cv2.putText(frame, name, (px + 5, py + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        bar_top = py + 26 + i * (BAR_H + BAR_GAP)
        filled  = int((PANEL_W - PAD * 2) * prob)
        ec = COLORS[emotion]

        cv2.rectangle(frame, (px + PAD, bar_top),
                      (px + PANEL_W - PAD, bar_top + BAR_H), (40, 40, 40), -1)
        if filled > 0:
            cv2.rectangle(frame, (px + PAD, bar_top),
                          (px + PAD + filled, bar_top + BAR_H), ec, -1)

        weight = 2 if i == top_idx else 1
        cv2.putText(frame, f'{emotion} {prob * 100:.0f}%',
                    (px + PAD + 3, bar_top + BAR_H - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), weight)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fh, fw = frame.shape[:2]
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    all_probs = None

    for (x, y, w, h) in faces:
        face_roi  = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
        all_probs = [fn(face_roi) for _, fn, _ in MODELS]

        # consensus = average of all 3 models' probability distributions
        consensus_probs = np.mean(all_probs, axis=0)
        top_idx  = int(np.argmax(consensus_probs))
        label    = f'{EMOTIONS[top_idx]}  {consensus_probs[top_idx] * 100:.0f}%'
        box_col  = COLORS[EMOTIONS[top_idx]]

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_col, 2)
        cv2.rectangle(frame, (x, y - 28), (x + w, y), box_col, -1)
        cv2.putText(frame, label, (x + 4, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    # fixed corner panels — only drawn when a face is detected
    if all_probs is not None:
        PANEL_H = 26 + len(EMOTIONS) * (BAR_H + BAR_GAP) + 4
        corners = [
            (PANEL_GAP, PANEL_GAP),                      # top-left
            (fw - PANEL_W - PANEL_GAP, PANEL_GAP),       # top-right
            (PANEL_GAP, fh - PANEL_H - 34),              # bottom-left
        ]
        for (px, py), (name, _, hcol), probs in zip(corners, MODELS, all_probs):
            draw_panel(frame, probs, name, hcol, px, py)

    # footer bar
    cv2.rectangle(frame, (0, fh - 28), (fw, fh), (20, 20, 20), -1)
    cv2.putText(frame, 'Face label = ensemble consensus (avg of all 3)        Q: quit',
                (10, fh - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)

    cv2.imshow('Emotion Recognition — Model Comparison', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
