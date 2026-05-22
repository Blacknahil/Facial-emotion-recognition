import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- Kaggle path ---
# DATA_PATH = '/kaggle/input/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013.csv'

# --- Local path (after downloading fer2013.csv into data/) ---
DATA_PATH = 'data/fer2013.csv'


def load_data(path):
    df = pd.read_csv(path)
    X = np.array([
        np.array(row.split(), dtype=np.float32).reshape(48, 48, 1) / 255.0
        for row in df['pixels']
    ])
    y = tf.keras.utils.to_categorical(df['emotion'], num_classes=7)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Accuracy over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Loss over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


def plot_confusion_matrix(model, X_val, y_val):
    y_pred = model.predict(X_val).argmax(axis=1)
    y_true = y_val.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTIONS, yticklabels=EMOTIONS)
    plt.title('Confusion Matrix — Which emotions get confused?')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    print(classification_report(y_true, y_pred, target_names=EMOTIONS))


if __name__ == '__main__':
    print("Loading data...")
    X_train, X_val, y_train, y_val = load_data(DATA_PATH)
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    model = build_model()
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
    )

    model.save('emotion_model.h5')
    print("Model saved to emotion_model.h5")

    plot_history(history)
    plot_confusion_matrix(model, X_val, y_val)
