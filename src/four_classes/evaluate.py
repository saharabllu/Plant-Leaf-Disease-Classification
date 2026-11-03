"""Evaluation metrics and visualization for the 4-Class model."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_4_classes(model, test_gen, history, save_cm_path: str = None):
    """Evaluates model, plots history, and generates Confusion Matrix."""
    
    # 1. Plot Training History
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train', color='#2196F3')
    axes[0].plot(history.history['val_accuracy'], label='Val', color='#4CAF50')
    axes[0].set_title('Akurasi Training vs Validasi')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train', color='#F44336')
    axes[1].plot(history.history['val_loss'], label='Val', color='#FF9800')
    axes[1].set_title('Loss Training vs Validasi')
    axes[1].legend()
    plt.show()

    # 2. Confusion Matrix
    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels
    )
    plt.title('Confusion Matrix - 4 Kelas')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_cm_path:
        plt.savefig(save_cm_path, dpi=150, bbox_inches='tight')
    plt.show()

    # 3. Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))
