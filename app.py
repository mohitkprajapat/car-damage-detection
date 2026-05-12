import os
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import uuid
from dotenv import load_dotenv
from src import config
from src.predictor import Predictor
from utils.utils import clear_old_uploads
from utils.auth import login_required, safe_compare, _set_role, LoginForm, limiter
from flask import Flask, redirect, render_template, request, url_for, session


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")
USER_PASSWORD = os.getenv("USER_LOGIN_PASSWORD")
load_dotenv()
UPLOAD_DIR = config.upload_path
os.makedirs(UPLOAD_DIR, exist_ok=True)

# load predictor once at startup
predictor = None
predictor_error = None

try:
    predictor = Predictor()
except FileNotFoundError as e:
    predictor_error = str(e)
except Exception as e:
    predictor_error = f"Failed to load model: {e}"


@app.route("/")
@login_required
def index():
    clear_old_uploads(UPLOAD_DIR)
    return render_template("index.html", error=predictor_error)


@app.route("/predict", methods=["POST"])
@login_required
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

    result = predictor.predict(img_path)

    return render_template(
        "result.html",
        uploaded=fname,
        pred_class=result["pred_class"],
        confidence=round(result["confidence"] * 100, 1),
        probs={k: round(v * 100, 1) for k, v in result["probs"].items()},
    )


@app.route("/about")
@login_required
def about():
    return render_template("index.html", show_about=True, error=predictor_error)


@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def login():
    form = LoginForm()
    if form.validate_on_submit():
        password = form.password.data
        if safe_compare(password, USER_PASSWORD):
            _set_role("user")
            return redirect(url_for("index"))
        else:
            return render_template("login.html", form=form, error="Invalid password")
    return render_template("login.html", form=form)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
