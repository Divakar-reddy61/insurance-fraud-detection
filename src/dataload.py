# src/dataload.py
from pathlib import Path
import pandas as pd

def load_dataset():
    """
    Load the CSV dataset safely and print full summary
    """
    # Project root = parent of src
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Dataset path
    csv_path = BASE_DIR / "dataset" / "raw" / "vehicle_damage_dataset" / "labels.csv"

    # Check file exists
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # -----------------------------
    # PRINT FULL SUMMARY
    # -----------------------------
    print("📄 CSV loaded successfully!")
    print("CSV path:", csv_path)
    print("Total rows:", len(df))
    print("\n🔹 HEAD")
    print(df.head())
    print("\n🔹 TAIL")
    print(df.tail())
    print("\n🔹 SHAPE")
    print(df.shape)
    print("\n🔹 INFO")
    print(df.info())
    print("\n🔹 DESCRIBE")
    print(df.describe(), "\n")

    return df, csv_path

# -----------------------------
# Optional: test if run directly
# -----------------------------
if __name__ == "__main__":
    load_dataset()
