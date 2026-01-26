# src/evaluate.py
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.model_selection import StratifiedKFold

def run_evaluation(
    base_dir=r"D:\INSURENCEFRAUDDETECTION",
    csv_file="dataset/processed/labels_no_blur.csv",
    image_folder="dataset/raw/vehicle_damage_dataset",
    model_file="saved_models/final_cnn_model.h5",
    img_size=224
):
    """
    Loads the saved CNN model and evaluates it on the dataset.
    Plots confusion matrix, ROC curve, and prints classification metrics.
    """
    RAW_DIR = os.path.join(base_dir, image_folder)
    CSV_PATH = os.path.join(base_dir, csv_file)
    MODEL_PATH = os.path.join(base_dir, model_file)

    # Load model
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully")

    # Load CSV & create labels
    df = pd.read_csv(CSV_PATH)
    df["label"] = df["image_path"].apply(lambda x: 1 if x.lower().startswith("real") else 0)
    print("✅ CSV loaded | Total samples:", len(df))

    # Load images
    X_test, y_test = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(RAW_DIR, row["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        X_test.append(img)
        y_test.append(row["label"])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print("✅ Test images loaded:", X_test.shape[0])

    # Model predictions
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Performance metrics
    print("\n📊 Classification Report")
    print(classification_report(y_test, y_pred))
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1], [0,1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Statistical Validation")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Stratified K-Fold consistency
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    acc_scores = []
    for _, test_idx in skf.split(X_test, y_test):
        y_fold_true = y_test[test_idx]
        y_fold_pred = (y_pred_prob[test_idx] > 0.5).astype(int)
        acc_scores.append(accuracy_score(y_fold_true, y_fold_pred))
    print("\n📊 Statistical Consistency Check")
    print("Accuracy across folds:", acc_scores)
    print("Mean Accuracy:", round(np.mean(acc_scores), 4))
    print("Std Deviation:", round(np.std(acc_scores), 4))

    return y_test, y_pred
if __name__ == "__main__":
    run_evaluation()
