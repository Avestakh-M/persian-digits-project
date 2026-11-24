import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from config import CM_PLOT_PATH, ACC_PLOT_PATH, PERSIAN_DIGITS

def plot_training_curves(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.title('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.title('Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(ACC_PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_model(model, test_ds):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=PERSIAN_DIGITS, yticklabels=PERSIAN_DIGITS)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(CM_PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.show()

    print(classification_report(y_true, y_pred,
                                target_names=PERSIAN_DIGITS, digits=4))
    return y_true, y_pred
