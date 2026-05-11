import os
import uuid
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
load_dotenv()
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# load predictor once at startup
predictor = None
predictor_error = None

try:
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from flask_app.predictor import Predictor

    predictor = Predictor()
except FileNotFoundError as e:
    predictor_error = str(e)
except Exception as e:
    predictor_error = f"Failed to load model: {e}"


@app.route("/")
def index():
    return render_template("index.html", error=predictor_error)


@app.route("/predict", methods=["POST"])
def predict():
    if predictor is None:
        return render_template("index.html", error=predictor_error)

    if "image" not in request.files or request.files["image"].filename == "":
        return redirect(url_for("index"))

    f = request.files["image"]
    ext = os.path.splitext(f.filename)[1].lower()
    fname = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    img_path = os.path.join(UPLOAD_DIR, fname)
    f.save(img_path)

    result = predictor.predict_with_gradcam(img_path, UPLOAD_DIR)

    return render_template(
        "result.html",
        uploaded=fname,
        pred_class=result["pred_class"],
        confidence=round(result["confidence"] * 100, 1),
        probs={k: round(v * 100, 1) for k, v in result["probs"].items()},
        gradcam=result["gradcam_path"],
    )


@app.route("/about")
def about():
    return render_template("index.html", show_about=True, error=predictor_error)


if __name__ == "__main__":
    app.run(debug=True)
