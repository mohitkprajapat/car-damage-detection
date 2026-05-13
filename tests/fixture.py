import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture(scope="session")
def app():
    """Create the Flask app once for the whole test session."""
    with patch("app.Predictor") as mock_pred_cls:
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {
            "pred_class": "minor",
            "confidence": 0.85,
            "probs": {"minor": 0.85, "moderate": 0.10, "severe": 0.05},
            "score": 3.1,
        }
        mock_pred_cls.return_value = mock_predictor

        from app import app as flask_app

        flask_app.config.update(
            TESTING=True,
            WTF_CSRF_ENABLED=False,
            SECRET_KEY="test-secret-key-32-chars-long!!",
        )
        yield flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def auth_client(client):
    """A test client that is already logged-in."""
    client.post("/login", data={"password": "testpassword123"})
    return client


@pytest.fixture()
def tmp_upload_dir(tmp_path):
    return tmp_path / "uploads"