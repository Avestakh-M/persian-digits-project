import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from config import IMG_SIZE, PERSIAN_DIGITS

def predict_digit(image_path, model):
    img = keras.utils.load_img(image_path, color_mode="grayscale", target_size=IMG_SIZE)
    arr = keras.utils.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds))
    conf = float(preds[0][idx] * 100)
    print(f"پیش‌بینی: {PERSIAN_DIGITS[idx]} | اطمینان: {conf:.2f}%")
    plt.figure(figsize=(3, 3)); plt.imshow(img, cmap='gray')
    plt.title(f"{PERSIAN_DIGITS[idx]} ({conf:.1f}%)"); plt.axis('off'); plt.show()
    return idx, conf
