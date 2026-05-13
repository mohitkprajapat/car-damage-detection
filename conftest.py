"""
conftest.py  (project root)
============================
Runs before pytest collects or imports anything.
Sets all required env vars using a single consistent Fernet key so that
utils/auth.py, app.py, and the tests all share the same key at import time.
"""
import sys
from pathlib import Path

# ── 1. Put project root on sys.path so src/, utils/, app.py are importable ──
sys.path.insert(0, str(Path(__file__).parent))

# ── 2. Set env vars BEFORE any app module is imported ────────────────────────
import os
from cryptography.fernet import Fernet

# Generate one key for the whole test session and share it everywhere.
_TEST_FERNET_KEY = Fernet.generate_key().decode()

os.environ["SESSION_FERNET_KEY"]  = _TEST_FERNET_KEY
os.environ["FLASK_SECRET_KEY"]    = "test-secret-key-32-chars-long!!"
os.environ["USER_LOGIN_PASSWORD"] = "testpassword123"
os.environ.setdefault("ROOT_DIR", str(Path(__file__).parent))