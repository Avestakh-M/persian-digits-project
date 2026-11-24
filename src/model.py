from tensorflow import keras
from tensorflow.keras import layers
from config import IMG_SIZE, NUM_CLASSES

def build_model():
    aug = keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.05),
    ])
    model = keras.Sequential([
        layers.Input(shape=(*IMG_SIZE, 1)),
        aug,
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model
