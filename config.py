# === مسیرها ===
DATA_ROOT   = "/content/data"                                       # ← از قدم ۲-۳
OUTPUT_ROOT = "/content/drive/MyDrive/persian-digits/outputs"

TRAIN_DIR = f"{DATA_ROOT}/Train"
TEST_DIR  = f"{DATA_ROOT}/Test"

MODEL_PATH    = f"{OUTPUT_ROOT}/best_model.keras"
HISTORY_PATH  = f"{OUTPUT_ROOT}/history.json"
CM_PLOT_PATH  = f"{OUTPUT_ROOT}/confusion_matrix.png"
ACC_PLOT_PATH = f"{OUTPUT_ROOT}/training_curves.png"

# === ابرپارامترها ===
IMG_SIZE      = (28, 28)
BATCH_SIZE    = 64
EPOCHS        = 30
NUM_CLASSES   = 10
LEARNING_RATE = 1e-3

PERSIAN_DIGITS = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹']
