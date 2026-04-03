from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Skin_Diseases.h5"
UPLOAD_DIR = BASE_DIR / "uploads"
IMG_SIZE = (64, 64)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
CLASS_NAMES = ["Acne", "Melanoma", "Psoriasis", "Rosacea", "Vitiligo"]

try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load_model(str(MODEL_PATH))
    MODEL_LOAD_ERROR = None
except Exception as exc:
    model = None
    MODEL_LOAD_ERROR = exc


def allowed_file(filename: str) -> bool:
    return "." in filename and Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return f"Model is not loaded: {MODEL_LOAD_ERROR}", 500

    uploaded_file = request.files.get("file")
    if not uploaded_file or uploaded_file.filename == "":
        return "No file uploaded", 400

    if not allowed_file(uploaded_file.filename):
        return "Unsupported file type", 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    filename = secure_filename(uploaded_file.filename)
    if filename == "":
        return "Invalid filename", 400
    file_path = UPLOAD_DIR / filename
    uploaded_file.save(str(file_path))

    try:
        # Image preprocessing
        img = load_img(str(file_path), target_size=IMG_SIZE)
        img_array = img_to_array(img).astype("float32")
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array, verbose=0)
        pred_index = int(np.argmax(predictions, axis=1)[0])
        prediction = CLASS_NAMES[pred_index]

        return f"Prediction: {prediction}"
    except Exception as exc:
        return f"Failed to process image: {exc}", 400
    finally:
        try:
            file_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
