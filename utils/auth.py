import hmac
import os
from functools import wraps

from cryptography.fernet import Fernet
from dotenv import load_dotenv
from flask import redirect, session, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import FlaskForm
from wtforms import PasswordField
from wtforms.validators import DataRequired, Length

load_dotenv()

_fernet = Fernet(os.getenv("SESSION_FERNET_KEY").encode())
limiter = Limiter(get_remote_address, default_limits=["500 per minute"], storage_uri="memory://")

def safe_compare(a, b):
    if not a or not b:
        return False
    return hmac.compare_digest(a.encode(), b.encode())


def _decrypt(token: str) -> str | None:
    try:
        return _fernet.decrypt(token.encode()).decode()
    except Exception:
        return None
    
def _encrypt(value: str) -> str:
    """Encrypt a session value so it is opaque even if the session is decoded."""
    return _fernet.encrypt(value.encode()).decode()

def _session_role() -> str | None:
    raw = session.get("role")
    return _decrypt(raw) if raw else None


def _integrity_valid() -> bool:
    sig = session.get("_sig")
    if not sig:
        return False

    role = _session_role()
    if role is None:
        return False

    expected = hmac.new(
        os.getenv("SESSION_FERNET_KEY").encode(), role.encode(), "sha256"
    ).hexdigest()
    return hmac.compare_digest(sig, expected)

def safe_compare(a, b):
    if not a or not b:
        return False
    return hmac.compare_digest(a.encode(), b.encode())

def _set_role(role: str):
    session.clear()
    session["role"] = _encrypt(role)
    session["_sig"] = hmac.new(
        os.getenv("SESSION_FERNET_KEY").encode(), role.encode(), "sha256"
    ).hexdigest()


class LoginForm(FlaskForm):
    password = PasswordField("Password", validators=[DataRequired(), Length(min=4, max=128)])


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not _integrity_valid():
            session.clear()
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function