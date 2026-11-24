import tensorflow as tf
from config import TRAIN_DIR, TEST_DIR, IMG_SIZE, BATCH_SIZE

def load_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        color_mode="grayscale", label_mode="categorical",
        shuffle=True, seed=42
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        color_mode="grayscale", label_mode="categorical", shuffle=False
    )
    norm = tf.keras.layers.Rescaling(1./255)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    return train_ds, test_ds
