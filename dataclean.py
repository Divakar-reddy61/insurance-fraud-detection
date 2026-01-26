import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import imagehash

# =================================================
# PATH SETUP (PORTABLE – WORKS ON ANY SYSTEM)
# =================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw", "vehicle_damage_dataset")
PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")

CSV_PATH = os.path.join(RAW_DIR, "labels.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# =================================================
# MAIN CLEANING FUNCTION
# =================================================
def clean_dataset():
    print("📂 RAW DIR       :", RAW_DIR)
    print("📂 PROCESSED DIR :", PROCESSED_DIR)
    print("📄 CSV PATH      :", CSV_PATH)

    # -----------------------------
    # Load CSV
    # -----------------------------
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print("\n✅ Step 0: Original dataset loaded")
    print("Total rows:", len(df))

    # -----------------------------
    # 1️⃣ Remove duplicate rows
    # -----------------------------
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"\n✅ Step 1: Duplicate rows removed | Before: {before}, After: {after}, Removed: {before - after}")

    # -----------------------------
    # 2️⃣ Remove rows with missing values
    # -----------------------------
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"\n✅ Step 2: Missing values removed | Before: {before}, After: {after}, Removed: {before - after}")

    # -----------------------------
    # 3️⃣ Remove missing image files
    # -----------------------------
    before = len(df)

    def image_exists(row):
        return os.path.isfile(os.path.join(RAW_DIR, row["image_path"]))

    df = df[df.apply(image_exists, axis=1)].reset_index(drop=True)
    after = len(df)
    print(f"\n✅ Step 3: Missing images removed | Before: {before}, After: {after}, Removed: {before - after}")

    # -----------------------------
    # 4️⃣ Remove corrupted images
    # -----------------------------
    before = len(df)
    valid_rows = []
    for _, row in df.iterrows():
        img_path = os.path.join(RAW_DIR, row["image_path"])
        if cv2.imread(img_path) is not None:
            valid_rows.append(row)

    df = pd.DataFrame(valid_rows)
    after = len(df)
    print(f"\n✅ Step 4: Corrupted images removed | Before: {before}, After: {after}, Removed: {before - after}")

    # -----------------------------
    # 5️⃣ Remove duplicate images (pHash)
    # -----------------------------
    before = len(df)
    hashes = {}
    unique_rows = []

    for _, row in df.iterrows():
        img_path = os.path.join(RAW_DIR, row["image_path"])
        try:
            img = Image.open(img_path).convert("RGB")
            img_hash = imagehash.phash(img)
        except:
            continue

        if img_hash not in hashes:
            hashes[img_hash] = img_path
            unique_rows.append(row)

    df = pd.DataFrame(unique_rows)
    after = len(df)
    print(f"\n✅ Step 5: Duplicate images (by hash) removed | Before: {before}, After: {after}, Removed: {before - after}")

    # -----------------------------
    # 6️⃣ Blur Detection
    # -----------------------------
    before = len(df)
    BLUR_THRESHOLD = 100
    clear_rows, blur_rows = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(RAW_DIR, row["image_path"])
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        if variance < BLUR_THRESHOLD:
            blur_rows.append(row)
        else:
            clear_rows.append(row)

    df_clear = pd.DataFrame(clear_rows)
    df_blur = pd.DataFrame(blur_rows)
    after = len(df_clear)
    print(f"\n✅ Step 6: Blur images removed | Before: {before}, After: {after}, Removed: {before - after}")
    print(f"⚠️ Blur images detected: {len(df_blur)}")

    # -----------------------------
    # Save cleaned CSV
    # -----------------------------
    clear_csv = os.path.join(PROCESSED_DIR, "labels_clean_final.csv")
    blur_csv = os.path.join(PROCESSED_DIR, "labels_blur.csv")
    df_clear.to_csv(clear_csv, index=False)
    df_blur.to_csv(blur_csv, index=False)

    print("\n📁 Clean CSV saved:", clear_csv)
    print("📁 Blur CSV saved:", blur_csv)
    print("\n🎯 Cleaning pipeline completed successfully!")

    return clear_csv

# =================================================
# SCRIPT ENTRY POINT
# =================================================
if __name__ == "__main__":
    output_csv = clean_dataset()
    print("🎯 Cleaning completed successfully!")
    print("📄 Output file:", output_csv)
