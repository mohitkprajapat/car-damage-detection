import os
import warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import uuid

from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for, send_file

from src import config
from src.predictor import Predictor
from utils.auth import LoginForm, _set_role, limiter, login_required, safe_compare
from utils.utils import clear_old_uploads, is_report_stale
from monitoring.data_collection import get_unlabeled_rows, log_prediction, set_true_label
from monitoring.monitoring import evi_monitor

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")
    limiter.init_app(app)
    return app

app = create_app()

USER_PASSWORD = os.getenv("USER_LOGIN_PASSWORD")
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
    unlabeled_files = {row["image_path"] for row in get_unlabeled_rows()}
    clear_old_uploads(UPLOAD_DIR, protect=unlabeled_files)
    return render_template("index.html", error=predictor_error)


@app.route("/predict", methods=["POST"])
@login_required
@limiter.limit("30 per day")
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
    log_prediction(result, fname)

    return render_template(
        "result.html",
        uploaded=fname,
        pred_class=result["pred_class"],
        confidence=round(result["confidence"] * 100, 1),
        probs={k: round(v * 100, 1) for k, v in result["probs"].items()},
        damage_score=result["score"],
    )


@app.route("/about")
@login_required
def about():
    return render_template("index.html", show_about=True, error=predictor_error)

@app.route("/review")
@login_required
def review():
    rows = get_unlabeled_rows()
    for row in rows:
        try:
            row["confidence_pct"] = round(float(row["confidence"]) * 100, 1)
        except (TypeError, ValueError):
            row["confidence_pct"] = None
    return render_template("review.html", rows=rows, class_labels=config.class_labels)


@app.route("/review/label", methods=["POST"])
@login_required
def review_label():
    image_path = request.form.get("image_path")
    true_label = request.form.get("true_label")
    if image_path and true_label:
        set_true_label(image_path, true_label)
    return redirect(url_for("review"))

@app.route("/monitor")
@login_required
def monitor():
    is_old = is_report_stale()
    if is_old:
        evi_monitor()
    return render_template("monitor.html")

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

@app.route("/nologin")
def nologin():
    _set_role("user")
    return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route('/favicon.ico')
def favicon():
    FEVICON_PATH = os.path.join(config.root_dir, "static", "images","favicon.png")
    return send_file(FEVICON_PATH, mimetype='image/png')

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=8080, debug=debug_mode)
