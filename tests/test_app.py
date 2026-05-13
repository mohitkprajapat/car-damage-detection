import io
from unittest.mock import patch

from tests.fixture import auth_client, app, client

def _make_image_bytes() -> bytes:
    """Return minimal valid PNG bytes (1*1 red pixel)."""
    import struct, zlib

    def png_chunk(name: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + name + data
        return c + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr = png_chunk(b"IHDR", ihdr_data)
    raw = b"\x00\xff\x00\x00"
    idat = png_chunk(b"IDAT", zlib.compress(raw))
    iend = png_chunk(b"IEND", b"")
    return signature + ihdr + idat + iend

class TestPredictRoute:
    def test_predict_no_file_redirects(self, auth_client):
        resp = auth_client.post("/predict", data={})
        assert resp.status_code == 302

    def test_predict_empty_filename_redirects(self, auth_client):
        data = {"image": (io.BytesIO(b""), "")}
        resp = auth_client.post(
            "/predict", data=data, content_type="multipart/form-data"
        )
        assert resp.status_code == 302

    def test_predict_valid_image(self, auth_client):
        png = _make_image_bytes()
        data = {"image": (io.BytesIO(png), "test_car.png")}
        resp = auth_client.post(
            "/predict", data=data, content_type="multipart/form-data"
        )
        assert resp.status_code == 200
        assert b"minor" in resp.data.lower()

    def test_predict_result_contains_confidence(self, app):
        """Verify confidence value appears in the result page."""
        with patch("app.predictor") as mock_p:
            mock_p.predict.return_value = {
                "pred_class": "minor",
                "confidence": 0.85,
                "probs": {"minor": 0.85, "moderate": 0.10, "severe": 0.05},
                "score": 3.1,
            }
            with app.test_client() as c:
                c.get("/nologin")
                png = _make_image_bytes()
                resp = c.post(
                    "/predict",
                    data={"image": (io.BytesIO(png), "car.png")},
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 200, f"Got {resp.status_code}, expected 200"
                assert b"85" in resp.data

    def test_predict_result_contains_score(self, app):
        """Verify damage score appears in the result page."""
        with patch("app.predictor") as mock_p:
            mock_p.predict.return_value = {
                "pred_class": "minor",
                "confidence": 0.85,
                "probs": {"minor": 0.85, "moderate": 0.10, "severe": 0.05},
                "score": 3.1,
            }
            
            with app.test_client() as c:
                c.get("/nologin")
                png = _make_image_bytes()
                resp = c.post(
                    "/predict",
                    data={"image": (io.BytesIO(png), "car.png")},
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 200
                assert b"3.1" in resp.data

    def test_predict_no_model_shows_error(self, app):
        """When predictor is None the page should display an error, not crash."""
        import app as app_mod

        original = app_mod.predictor
        original_err = app_mod.predictor_error
        app_mod.predictor = None
        app_mod.predictor_error = "No model found."

        try:
            with app.test_client() as c:
                c.get("/nologin")
                png = _make_image_bytes()
                resp = c.post(
                    "/predict",
                    data={"image": (io.BytesIO(png), "car.png")},
                    content_type="multipart/form-data",
                )
                assert resp.status_code == 200, f"Got {resp.status_code}, expected 200"
                assert b"model" in resp.data.lower()
        finally:
            app_mod.predictor = original
            app_mod.predictor_error = original_err