
from db_operations import save_prediction, fetch_predictions
import os
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, session
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)

# ------------------ CONFIG ------------------
UPLOAD_FOLDER = "static/uploads"
FEATURE_FOLDER = "static/feature_maps"
CNN_MODEL_PATH = "saved_models/final_cnn_model.h5"
IMG_SIZE = 224
SECRET_KEY = "your-secret-key"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FEATURE_FOLDER, exist_ok=True)

# ------------------ FLASK APP ------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

# ------------------ LOAD CNN MODEL ------------------
cnn_model = load_model(CNN_MODEL_PATH)
print("✅ CNN Model loaded successfully")

# ------------------ FEATURE MAP MODEL ------------------
conv_layers = [layer.output for layer in cnn_model.layers if "conv" in layer.name.lower()]
feature_model = Model(inputs=cnn_model.input, outputs=conv_layers)

# ------------------ LAYER DESCRIPTIONS ------------------
LAYER_DESCRIPTIONS = {
    "conv_layer_1": "Detects basic edges, contours, and structural outlines of the vehicle.",
    "conv_layer_2": "Extracts texture inconsistencies, cracks, lighting variations, and shadows.",
    "conv_layer_3": "Identifies complex damage regions and synthetic manipulation artifacts."
}

# ------------------ AUTO EXPLANATION ------------------
def generate_explanation(pred):
    explanation = ["Input image resized to 224×224 and normalized."]
    if pred >= 0.7:
        explanation += [
            "Strong structural consistency detected.",
            "Damage patterns match real accident images.",
            "High confidence in image authenticity."
        ]
    elif pred >= 0.5:
        explanation += [
            "Moderate consistency in damage patterns.",
            "Some ambiguity detected in texture regions."
        ]
    else:
        explanation += [
            "Irregular textures and unnatural edges detected.",
            "High probability of image manipulation."
        ]
    explanation.append("CNN combined multi-level features to make final decision.")
    return explanation

# ------------------ SAVE FEATURE MAPS ------------------
def save_feature_maps(feature_maps):
    saved_images = {}
    for idx, fmap in enumerate(feature_maps):
        layer_name = f"conv_layer_{idx + 1}"
        layer_dir = os.path.join(FEATURE_FOLDER, layer_name)
        os.makedirs(layer_dir, exist_ok=True)

        saved_images[layer_name] = []
        for i in range(min(6, fmap.shape[-1])):
            img = fmap[0, :, :, i]
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img_path = f"{layer_dir}/{layer_name}_{i}.png"
            plt.imsave(img_path, img, cmap="gray")
            saved_images[layer_name].append(img_path)

    return saved_images

# ------------------ VEHICLE CHECK (MobileNetV2) ------------------
vehicle_model = MobileNetV2(weights="imagenet", include_top=True)




def is_vehicle(img_path, threshold=0.15):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = vehicle_model.predict(img_array)
    decoded = decode_predictions(preds, top=10)[0]

    VEHICLE_KEYWORDS = [
        "car", "jeep", "cab", "taxi",
        "truck", "bus", "pickup",
        "van", "minivan",
        "bumper", "grille", "headlight",
        "automobile", "sedan"
    ]

    for _, label, prob in decoded:
        if any(k in label.lower() for k in VEHICLE_KEYWORDS) and prob >= threshold:
            return True, label, float(prob)

    return False, None, None


# ------------------ HOME PAGE ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_path = None
    explanation = []
    feature_images = {}
    invalid_message = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(img_path)

            # Step 1: Vehicle Validation
            is_valid, vehicle_label, vehicle_conf = is_vehicle(img_path)

            if not is_valid:
                invalid_message = "⚠️ Please upload a valid vehicle damage image."
            else:
                img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                pred = float(cnn_model.predict(img_array)[0][0])
                authenticity = round(pred * 100, 2)
                manipulation = round((1 - pred) * 100, 2)

                result = {
                    "label": "REAL" if pred >= 0.5 else "FAKE",
                    "authenticity": authenticity,
                    "manipulation": manipulation,
                    "vehicle": vehicle_label,
                    "vehicle_confidence": round(vehicle_conf * 100, 2)
                }

                # Save to database
                try:
                    save_prediction(
                        image_name=file.filename,
                        prediction=result["label"],
                        confidence=authenticity
                    )
                except Exception as e:
                    print("Database Error:", e)

                # Feature maps
                feature_maps = feature_model.predict(img_array)
                feature_images = save_feature_maps(feature_maps)

                # Explanation
                explanation = generate_explanation(pred)

                # Save session data
                session["latest_uploaded_image"] = file.filename
                session["latest_confidence"] = authenticity
                session["latest_explanation"] = explanation
                session["latest_feature_images"] = feature_images

    return render_template(
        "index.html",
        result=result,
        img_path=img_path,
        explanation=explanation,
        feature_images=feature_images,
        layer_descriptions=LAYER_DESCRIPTIONS,
        invalid_message=invalid_message
    )

# ------------------ DASHBOARD PAGE ------------------
@app.route("/dashboard")
def dashboard():
    history = fetch_predictions()

    real_count = sum(1 for row in history if row[2] == "REAL")
    fake_count = sum(1 for row in history if row[2] == "FAKE")

    confidences = [float(row[3]) for row in history]
    time_labels = [row[4] for row in history]

    real_trend, fake_trend = [], []
    real_total = fake_total = 0

    for row in history:
        if row[2] == "REAL":
            real_total += 1
        else:
            fake_total += 1
        real_trend.append(real_total)
        fake_trend.append(fake_total)

    uploaded_image_url = (
        f"/static/uploads/{session.get('latest_uploaded_image')}"
        if session.get("latest_uploaded_image")
        else None
    )

    return render_template(
        "dashboard.html",
        history=history,
        real_count=real_count,
        fake_count=fake_count,
        confidences=confidences,
        time_labels=time_labels,
        real_trend=real_trend,
        fake_trend=fake_trend,
        uploaded_image_url=uploaded_image_url,
        confidence_score=session.get("latest_confidence"),
        explanation_text=session.get("latest_explanation", ["No explanation available"]),
        feature_images=session.get("latest_feature_images", {}),
        layer_descriptions=LAYER_DESCRIPTIONS
    )

# ------------------ DATABASE PAGE ------------------
@app.route("/database")
def database():
    history = fetch_predictions()
    return render_template("database.html", history=history)

# ------------------ MAIN ------------------
if __name__ == "__main__":
    app.run(debug=True)
