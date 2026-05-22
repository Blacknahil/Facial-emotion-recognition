import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = 'emotion_model.h5'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
COLORS = {
    'Angry':    (0,   0,   255),
    'Disgust':  (0,   140, 0),
    'Fear':     (128, 0,   128),
    'Happy':    (0,   255, 255),
    'Sad':      (255, 100, 0),
    'Surprise': (0,   165, 255),
    'Neutral':  (200, 200, 200),
}

model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def draw_emotion_bars(frame, probs, x, y, w):
    bar_x = x
    bar_y = y + 10
    bar_width = w
    bar_height = 14
    gap = 4

    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        filled = int(bar_width * prob)
        color = COLORS[emotion]
        top = bar_y + i * (bar_height + gap)

        cv2.rectangle(frame, (bar_x, top), (bar_x + bar_width, top + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, top), (bar_x + filled, top + bar_height), color, -1)
        label = f'{emotion}: {prob*100:.0f}%'
        cv2.putText(frame, label, (bar_x + bar_width + 6, top + bar_height - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)


cap = cv2.VideoCapture(0)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48)).astype(np.float32) / 255.0
        roi = roi.reshape(1, 48, 48, 1)

        probs = model.predict(roi, verbose=0)[0]
        top_idx = np.argmax(probs)
        emotion = EMOTIONS[top_idx]
        confidence = probs[top_idx]
        color = COLORS[emotion]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label = f'{emotion}  {confidence*100:.0f}%'
        cv2.rectangle(frame, (x, y-28), (x+w, y), color, -1)
        cv2.putText(frame, label, (x+4, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        bar_panel_x = x + w + 10
        if bar_panel_x + 160 < frame.shape[1]:
            draw_emotion_bars(frame, probs, bar_panel_x, y, 100)

    cv2.imshow('Emotion Recognition — Cognitive Science Project', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
