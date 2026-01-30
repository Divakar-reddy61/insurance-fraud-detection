# Insurance Fraud Detection - Vehicle Damage Analysis

A deep learning-based system for detecting fraudulent vehicle damage claims by analyzing images using CNN models. This project uses image classification to distinguish between real damage (authentic claims) and fake/manipulated images (potential fraud).

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Processing Pipeline](#data-processing-pipeline)
  - [Model Training](#model-training)
  - [Running the Flask Web App](#running-the-flask-web-app)
- [Dataset](#dataset)
- [Technologies](#technologies)
- [Model Architecture](#model-architecture)
- [API Endpoints](#api-endpoints)
- [Database Schema](#database-schema)
- [Results & Evaluation](#results--evaluation)
- [Contributing](#contributing)

---

## 🎯 Project Overview

Insurance fraud detection through vehicle damage image analysis involves:

1. **Data Cleaning & Preprocessing**: Remove corrupted images, duplicates, and blurry images
2. **Exploratory Data Analysis (EDA)**: Analyze image characteristics and distributions
3. **Feature Extraction**: Extract features using CNN layers
4. **Model Training**: Train a custom CNN to classify real vs. fake damage
5. **Web Interface**: Interactive dashboard for predictions and visualization
6. **Database Integration**: Store predictions and analysis results

The system helps insurance companies reduce fraud by automating the verification of vehicle damage claims.

---

## ✨ Features

- **Automated Image Cleaning**: Removes duplicates using perceptual hashing, detects blurry images, validates image integrity
- **Exploratory Data Analysis**: Class distribution, image dimensions, brightness analysis, corruption detection
- **Custom CNN Model**: Built from scratch for vehicle damage classification
- **Feature Visualization**: Displays convolutional layer outputs to explain model decisions
- **Web Dashboard**: User-friendly Flask interface for uploading and analyzing images
- **Database Storage**: MySQL integration for storing predictions and claim data
- **Explainability**: Automatic explanations based on prediction confidence

---

## 📁 Project Structure

```
INSURENCEFRAUDDETECTION/
├── app.py                          # Flask web application
├── main.py                         # Main pipeline orchestrator
├── requirements.txt                # Python dependencies
├── db_config.py                    # Database configuration
├── db_operations.py                # Database CRUD operations
├── test_mysql.py                   # Database connection test
│
├── src/                            # Source code directory
│   ├── dataload.py                # Load raw dataset
│   ├── dataclean.py               # Data cleaning operations
│   ├── EDA.py                     # Exploratory Data Analysis
│   ├── train.py                   # Model training pipeline
│   ├── evaluate.py                # Model evaluation metrics
│   ├── predict.py                 # Prediction utilities (empty)
│   └── __init__.py
│
├── notebook/                       # Jupyter notebooks
│   ├── dataclean.ipynb            # Data cleaning notebook
│   ├── dataload.ipynb             # Data loading notebook
│   ├── datapreprocessing.ipynb    # Preprocessing notebook
│   └── train_test.ipynb           # Training & testing notebook
│
├── dataset/                        # Dataset directory
│   ├── raw/                       # Raw dataset
│   │   └── vehicle_damage_dataset/
│   │       ├── real/              # Real damage images
│   │       ├── fake/              # Fake/manipulated images
│   │       ├── labels.csv         # Original labels
│   │       ├── clean_labels_final.csv
│   │       ├── labels_no_blur.csv
│   │       └── ... (other processed labels)
│   │
│   └── processed/                 # Processed/cleaned datasets
│       ├── labels_no_duplicates.csv
│       ├── labels_no_blur.csv
│       ├── labels_no_corrupted.csv
│       └── labels_clean.csv
│
├── saved_models/                   # Trained models
│   ├── best_cnn_model.h5          # Best model during training
│   └── final_cnn_model.h5         # Final trained model
│
├── static/                         # Flask static files
│   ├── uploads/                   # User-uploaded images
│   ├── image/                     # Sample images
│   └── feature_maps/              # Convolutional layer visualizations
│       ├── conv_layer_1/
│       ├── conv_layer_2/
│       └── conv_layer_3/
│
└── templates/                      # Flask HTML templates
    ├── index.html                 # Home page
    ├── dashboard.html             # Prediction dashboard
    └── database.html              # Database records view
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8+
- MySQL Server (for database functionality)
- Git

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd INSURENCEFRAUDDETECTION
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Database

Edit `db_config.py` with your MySQL credentials:

```python
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="your_username",
        password="your_password",
        database="insurance_db"
    )
```

### Step 5: Test Database Connection

```bash
python test_mysql.py
```

---

## 🚀 Usage

### Data Processing Pipeline

#### Full Pipeline Execution

Run the complete pipeline from data loading to model evaluation:

```bash
python main.py
```

This will:
1. Load the raw dataset
2. Clean and preprocess images
3. Run exploratory data analysis
4. Train the CNN model
5. Evaluate model performance

#### Individual Steps

**Data Loading:**
```python
from src.dataload import load_dataset
df, csv_path = load_dataset()
```

**Data Cleaning:**
```python
from src.dataclean import clean_dataset
clean_csv_path = clean_dataset()
```

**Run EDA:**
```python
from src.EDA import run_eda
run_eda()
```

### Model Training

```bash
python -c "from src.train import train_cnn_pipeline; train_cnn_pipeline()"
```

Or in Python:
```python
from src.train import train_cnn_pipeline
train_cnn_pipeline()
```

### Running the Flask Web App

```bash
python app.py
```

The web application will start at `http://localhost:5000`

**Features:**
- Upload vehicle damage images
- Get real-time predictions (Real vs. Fake)
- Visualize convolutional layer feature maps
- View prediction explanations
- Browse stored predictions in database

---

## 📊 Dataset

### Source Structure

```
dataset/raw/vehicle_damage_dataset/
├── real/vehicle_damage/REAL/         # Authentic damage images
└── fake/vehicle_damage/FAKE/         # Manipulated/fake images
```

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Images | ~10,000+ |
| Real Images | ~50% |
| Fake Images | ~50% |
| Image Size | 224×224 (processed) |
| Format | PNG/JPG |

### Data Cleaning Steps

1. **Remove Duplicates**: Uses perceptual hashing (imagehash.phash)
2. **Remove Corrupted Images**: Validates image integrity with PIL
3. **Remove Blurry Images**: Laplacian variance threshold (< 100)
4. **Verify Paths**: Ensures all image files exist

---

## 📥 Downloads

### Dataset

- **Vehicle Damage Dataset (Google Drive)**: [Download dataset](https://drive.google.com/drive/folders/1sCAj8d_CnVjKnXkuurI1e9nef3nJT-Z7?usp=drive_link)
  - Contains real and fake vehicle damage images organized under `real/` and `fake/` folders
  - Size: (please verify in Drive)
  - Usage: place the unzipped dataset under `dataset/raw/vehicle_damage_dataset/` to match project paths

### Pre-trained Models (optional)

- **Best CNN Model**: stored locally in `saved_models/best_cnn_model.h5` (not tracked in repo)
- **Final CNN Model**: stored locally in `saved_models/final_cnn_model.h5` (not tracked in repo)

## 🛠 Technologies

| Technology | Purpose |
|-----------|---------|
| **TensorFlow/Keras** | Deep learning framework |
| **OpenCV** | Image processing |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical computations |
| **Scikit-learn** | ML utilities & metrics |
| **Flask** | Web framework |
| **MySQL** | Database |
| **Matplotlib/Seaborn** | Visualization |
| **PIL/Pillow** | Image operations |
| **imagehash** | Duplicate detection |

---

## 🧠 Model Architecture

### CNN Model

```
Sequential Model:
├── Conv2D(32, 3×3, ReLU)
├── MaxPooling2D(2×2)
├── Conv2D(64, 3×3, ReLU)
├── MaxPooling2D(2×2)
├── Conv2D(128, 3×3, ReLU)
├── MaxPooling2D(2×2)
├── Flatten()
├── Dense(256, ReLU) + Dropout(0.5)
├── Dense(128, ReLU) + Dropout(0.5)
└── Dense(1, Sigmoid)  # Binary classification
```

### Training Configuration

- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Batch Size**: 32
- **Epochs**: 10 (with early stopping)
- **Data Split**: 70% train, 15% validation, 15% test

---

## 🌐 API Endpoints

### Flask Routes

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/dashboard` | GET | Prediction dashboard |
| `/predict` | POST | Upload image and get prediction |
| `/database` | GET | View all predictions |
| `/get_feature_maps/<image_id>` | GET | Get feature visualizations |

### Prediction Response

```json
{
  "prediction": 0.85,
  "class": "REAL",
  "confidence": "85%",
  "explanation": [
    "Input image resized to 224×224 and normalized.",
    "Strong structural consistency detected.",
    "Damage patterns match real accident images.",
    "High confidence in image authenticity.",
    "CNN combined multi-level features to make final decision."
  ],
  "feature_maps": {
    "conv_layer_1": ["static/feature_maps/conv_layer_1/0.png", ...],
    "conv_layer_2": [...],
    "conv_layer_3": [...]
  }
}
```

---

## 🗄 Database Schema

### Table: `predictions`

```sql
CREATE TABLE predictions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  image_name VARCHAR(255) NOT NULL,
  prediction_value FLOAT NOT NULL,
  predicted_class VARCHAR(50) NOT NULL,
  confidence FLOAT NOT NULL,
  upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  explanation TEXT
);
```

### Database Operations

- **Save Prediction**: `save_prediction(image_name, pred, class_label, confidence)`
- **Fetch Predictions**: `fetch_predictions(limit=100)`
- **Database Connection**: `get_db_connection()`

---

## 📈 Results & Evaluation

### Model Performance Metrics

The model is evaluated using:

- **Accuracy**: Overall correctness
- **Precision**: True positive rate among positives
- **Recall**: Detection rate of actual frauds
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: TP, TN, FP, FN distribution

### Evaluation Report

Run evaluation:

```bash
python -c "from src.evaluate import run_evaluation; run_evaluation()"
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Project Workflow

### Typical User Journey

1. **Start Application**
   ```bash
   python app.py
   ```

2. **Upload Vehicle Damage Image**
   - Navigate to `/dashboard`
   - Upload image (JPG/PNG)

3. **Get Prediction**
   - Model processes image
   - Returns classification: REAL or FAKE
   - Shows confidence level (0-100%)

4. **View Feature Maps**
   - Displays layer activations
   - Explains what model detected

5. **Review in Database**
   - View historical predictions
   - Analyze fraud patterns
   - Export reports

---

## ⚠️ Important Notes

- **Database Password**: Change default credentials in `db_config.py`
- **Model Path**: Update paths if using different directory structures
- **Image Size**: Model expects 224×224 images; preprocessing handles resizing
- **GPU Support**: Install `tensorflow-gpu` for faster training if GPU available

---

## 📞 Support & Contact

For issues or questions:
- Review the Jupyter notebooks for detailed examples
- Check the `src/` modules for function documentation
- Ensure MySQL server is running before using database features

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🔍 Key Features Summary

| Feature | Status |
|---------|--------|
| Data cleaning pipeline | ✅ Complete |
| EDA & visualization | ✅ Complete |
| CNN model training | ✅ Complete |
| Model evaluation | ✅ Complete |
| Flask web interface | ✅ Complete |
| Database integration | ✅ Complete |
| Feature visualization | ✅ Complete |
| Prediction explanations | ✅ Complete |
| API endpoints | ✅ Complete |

---

**Last Updated**: January 2026

**Project Status**: Active Development
