import hmac
import os
import time
from functools import wraps
import hashlib
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from flask import redirect, session, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import FlaskForm
from wtforms import PasswordField
from wtforms.validators import DataRequired, Length

load_dotenv()

limiter = Limiter(get_remote_address, default_limits=["500 per minute"], storage_uri="memory://")
SESSION_MAX_AGE = 7 * 24 * 60 * 60

_SESSION_KEY = os.getenv("SESSION_FERNET_KEY")
if not _SESSION_KEY:
    raise RuntimeError("SESSION_FERNET_KEY is not set")
_SESSION_KEY_BYTES = _SESSION_KEY.encode()

_fernet = Fernet(_SESSION_KEY_BYTES)

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
        _SESSION_KEY_BYTES, 
        role.encode(), 
        digestmod=hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(sig, expected)


def _session_expired() -> bool:
    login_time = session.get("_login_time")
    if login_time is None:
        return True
    return (time.time() - login_time) > SESSION_MAX_AGE

def _set_role(role: str):
    session.clear()
    session["role"] = _encrypt(role)
    session["_sig"] = hmac.new(
        _SESSION_KEY_BYTES, 
        role.encode(), 
        digestmod=hashlib.sha256
    ).hexdigest()
    session["_login_time"] = time.time()


class LoginForm(FlaskForm):
    password = PasswordField("Password", validators=[DataRequired(), Length(min=4, max=128)])


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not _integrity_valid() or _session_expired():
            session.clear()
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function