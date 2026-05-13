import time
from tests.fixture import auth_client, app, client


class TestAuthHelpers:
    def test_safe_compare_equal(self):
        from utils.auth import safe_compare
        assert safe_compare("abc123", "abc123") is True

    def test_safe_compare_unequal(self):
        from utils.auth import safe_compare
        assert safe_compare("abc123", "wrong") is False

    def test_safe_compare_empty(self):
        from utils.auth import safe_compare
        assert safe_compare("", "abc") is False
        assert safe_compare("abc", "") is False
        assert safe_compare("", "") is False

    def test_encrypt_decrypt_roundtrip(self):
        from utils.auth import _decrypt, _encrypt
        original = "user"
        assert _decrypt(_encrypt(original)) == original

    def test_decrypt_garbage_returns_none(self):
        from utils.auth import _decrypt
        assert _decrypt("not-a-valid-token") is None

    def test_set_role_writes_encrypted_value(self, client):
        """_set_role should store an encrypted, non-plaintext role in the session."""
        from utils.auth import _set_role, _decrypt

        with client.application.test_request_context():
            from flask import session
            _set_role("user")
            encrypted = session.get("role")
            assert encrypted is not None
            assert encrypted != "user"
            assert _decrypt(encrypted) == "user"

    def test_session_expired_old_time(self):
        """_session_expired should flag a login_time 8 days in the past."""
        import utils.auth as auth_mod
        from flask import Flask, session

        tmp_app = Flask(__name__)
        tmp_app.secret_key = "test-secret-for-session"
        ctx = tmp_app.test_request_context()
        ctx.push()
        try:
            session["_login_time"] = time.time() - (8 * 24 * 3600)
            assert auth_mod._session_expired() is True
        finally:
            ctx.pop()

    def test_session_not_expired_fresh(self):
        import utils.auth as auth_mod
        from flask import Flask, session

        tmp_app = Flask(__name__)
        tmp_app.secret_key = "test-secret-for-session"
        ctx = tmp_app.test_request_context()
        ctx.push()
        try:
            session["_login_time"] = time.time()
            assert auth_mod._session_expired() is False
        finally:
            ctx.pop()


class TestLoginLogout:
    def test_get_login_page(self, client):
        resp = client.get("/login")
        assert resp.status_code == 200
        assert b"password" in resp.data.lower()

    def test_login_wrong_password(self, client):
        resp = client.post("/login", data={"password": "wrongpassword"})
        assert resp.status_code == 200
        assert b"invalid password" in resp.data.lower()

    def test_login_correct_password(self, client):
        resp = client.post(
            "/login", data={"password": "testpassword123"}, follow_redirects=True
        )
        assert resp.status_code == 200

    def test_logout_clears_session(self, auth_client):
        resp = auth_client.get("/logout", follow_redirects=True)
        assert resp.status_code == 200
        resp2 = auth_client.get("/")
        assert resp2.status_code == 302
        assert "/login" in resp2.headers["Location"]

    def test_nologin_sets_session(self, client):
        resp = client.get("/nologin", follow_redirects=True)
        assert resp.status_code == 200


class TestRouteProtection:
    def test_index_redirects_unauthenticated(self, client):
        resp = client.get("/")
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]

    def test_predict_redirects_unauthenticated(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 302

    def test_about_redirects_unauthenticated(self, client):
        resp = client.get("/about")
        assert resp.status_code == 302

    def test_index_accessible_when_authenticated(self, auth_client):
        resp = auth_client.get("/")
        assert resp.status_code == 200

    def test_about_accessible_when_authenticated(self, auth_client):
        resp = auth_client.get("/about")
        assert resp.status_code == 200