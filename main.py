
import sys
from pathlib import Path

# ------------------------------
# Add src/ to Python path
# ------------------------------
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))


# ------------------------------
# Import functions
# ------------------------------
from dataload import load_dataset        # dataload.py function
from dataclean import clean_dataset      # dataclean.py function
from EDA import run_eda                  # eda.py function
from train import train_cnn_pipeline     # ✅ train.py function (ADDED)
from evaluate import run_evaluation

# ------------------------------
# Main pipeline
# ------------------------------
def main():
    print("🚀 MAIN PIPELINE STARTED\n")

    # 1️⃣ Load dataset (optional, just to check CSV exists)
    print("📥 Loading dataset using dataload.py ...")
    df, csv_path = load_dataset()
    print(f"✅ Dataset loaded from {csv_path}, total rows: {len(df)}\n")

    # 2️⃣ Clean dataset
    print("🧹 Cleaning dataset using dataclean.py ...")
    clean_csv_path = clean_dataset()
    print(f"✅ Cleaned CSV saved at: {clean_csv_path}\n")

    # 3️⃣ Run EDA
    print("📊 Running Exploratory Data Analysis (EDA) ...")
    run_eda()  # Automatically detects INSURENCEFRAUDDETECTION folder

    # 4️⃣ Train CNN Model
    print("\n🧠 Training CNN model ...")
    train_cnn_pipeline()

    print("📊 Running Model Evaluation ...")
    run_evaluation()

    

    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY")

# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    main()

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = "saved_models/final_cnn_model.h5"
THRESHOLD = 0.4

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")


def predict_image(image_path):
    if not os.path.exists(image_path):
        print("❌ Image path does not exist")
        return

    # Load & preprocess image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    prob_real = float(model.predict(img_array)[0][0])
    prob_fake = 1.0 - prob_real

    # Class decision
    if prob_real >= THRESHOLD:
        label = "REAL"
        raw_conf = prob_real
    else:
        label = "FAKE"
        raw_conf = prob_fake

    # Soft confidence (distance from decision boundary)
    confidence = abs(prob_real - THRESHOLD) / max(THRESHOLD, 1 - THRESHOLD)
    confidence = max(min(confidence, 0.99), 0.01)

    print("\n🖼️ Image Tested :", image_path)
    print("P(REAL)        :", round(prob_real, 4))
    print("P(FAKE)        :", round(prob_fake, 4))
    print("Prediction     :", label)
    print("Confidence     :", round(confidence, 4))


    


test_image_path = r"D:\CARINSURENCEDETECTION\dataset\raw\vehicle_damage_dataset\fake\vehicle_damage\FAKE\fake_car_damage_494.png"
predict_image(test_image_path)