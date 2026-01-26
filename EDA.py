import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# =================================================
# EDA FUNCTION
# =================================================
def run_eda():
    """
    Run exploratory data analysis on the processed dataset.
    Prints dataset info, plots class distribution, image dimensions,
    sample images, edge detection, texture, lighting, and correlation.
    Automatically detects project folder INSURENCEFRAUDDETECTION.
    """

    # -----------------------------
    # Automatic path setup
    # -----------------------------
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_DIR)  # Project root
    PROJECT_NAME = "INSURENCEFRAUDDETECTION"

    # Check if folder name matches
    if PROJECT_NAME not in BASE_DIR:
        print(f"⚠️ Warning: Project folder does not match '{PROJECT_NAME}'")
    
    RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw", "vehicle_damage_dataset")
    PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
    CSV_PATH = os.path.join(PROCESSED_DIR, "labels_no_blur.csv")

    print(f"📂 RAW DIR       : {RAW_DIR}")
    print(f"📂 PROCESSED DIR : {PROCESSED_DIR}")
    print(f"📄 CSV PATH      : {CSV_PATH}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    # -----------------------------
    # Load CSV
    # -----------------------------
    df = pd.read_csv(CSV_PATH)
    print("\n---EXPLORATORY DATA ANALYSIS-------")
    print("✅ CSV loaded")
    print("Total images:", len(df))

    # -----------------------------
    # Class column
    # -----------------------------
    df['class'] = df['image_path'].apply(
        lambda x: 'real' if x.lower().startswith('real') else 'fake'
    )

    # -----------------------------
    # Class distribution plot
    # -----------------------------
    class_counts = df['class'].value_counts()
    plt.figure()
    class_counts.plot(kind='bar')
    plt.title("Class Distribution (Real vs Fake)")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.show()

    # -----------------------------
    # Image dimensions
    # -----------------------------
    heights, widths = [], []
    for path in df['image_path'].sample(min(200, len(df))):  # sample max 200 for speed
        img_path = os.path.join(RAW_DIR, path)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        heights.append(h)
        widths.append(w)

    print("Height range:", min(heights), "-", max(heights))
    print("Width range :", min(widths), "-", max(widths))

    plt.figure()
    plt.hist(heights, bins=20)
    plt.title("Image Height Distribution")
    plt.xlabel("Height (pixels)")
    plt.ylabel("Count")
    plt.show()

    plt.figure()
    plt.hist(widths, bins=20)
    plt.title("Image Width Distribution")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Count")
    plt.show()

    # -----------------------------
    # Sample images
    # -----------------------------
    def show_samples(label, n=5):
        samples = df[df['class'] == label].head(n)
        plt.figure(figsize=(15,3))
        shown = 0
        for _, row in samples.iterrows():
            img_path = os.path.join(RAW_DIR, row['image_path'])
            if not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except:
                continue
            plt.subplot(1, n, shown + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(label.capitalize())
            shown += 1
            if shown == n:
                break
        plt.show()

    show_samples("real", 5)
    show_samples("fake", 5)

    # -----------------------------
    # Edge detection & texture for one real image
    # -----------------------------
    row = df[df['class'] == 'real'].iloc[0]
    img_path = os.path.join(RAW_DIR, row['image_path'])
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image (REAL)")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(edges, cmap="gray")
    plt.title("Edge / Crack Detection")
    plt.axis("off")
    plt.show()

    # Texture consistency
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_strength = laplacian.var()
    print("Texture variance:", texture_strength)
    plt.figure(figsize=(6,4))
    plt.imshow(laplacian, cmap="gray")
    plt.title("Texture Consistency")
    plt.axis("off")
    plt.show()

    # Lighting / Shadow
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = hsv[:,:,2]
    plt.figure(figsize=(6,4))
    plt.imshow(brightness, cmap="gray")
    plt.title("Lighting & Shadow Patterns")
    plt.axis("off")
    plt.show()

    # Manipulation artifacts
    noise = cv2.absdiff(gray, cv2.GaussianBlur(gray, (5,5), 0))
    plt.figure(figsize=(6,4))
    plt.imshow(noise, cmap="gray")
    plt.title("Manipulation Artifacts")
    plt.axis("off")
    plt.show()

    # -----------------------------
    # Feature extraction & correlation
    # -----------------------------
    def blur_variance(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    data = []
    for _, row in df.iterrows():
        img_path = os.path.join(RAW_DIR, row["image_path"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        data.append({
            "width": w,
            "height": h,
            "pixel_mean": img.mean(),
            "pixel_std": img.std(),
            "brightness": hsv[:,:,2].mean(),
            "blur_var": blur_variance(img),
            "label": 1 if row["class"]=="real" else 0
        })

    stats_df = pd.DataFrame(data)
    print("✅ Features extracted:", stats_df.shape)

    # Correlation heatmap
    corr = stats_df.corr(numeric_only=True)
    plt.figure(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Analysis – Vehicle Damage Dataset")
    plt.show()

    print("🎯 EDA Completed Successfully!")
if __name__ == "__main__":
    run_eda() 