from __future__ import annotations

import io
import logging
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as api


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


def make_image_bytes(image_format: str) -> bytes:
    image = Image.new("RGB", (4, 4), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


class AnalyzeImageMock:
    def __init__(
        self,
        *,
        response: dict | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response or {
            "image_name": "uploaded.jpg",
            "detection_count": 1,
            "detections": [
                {
                    "label": "knife",
                    "score": 0.91,
                    "box": [1.0, 2.0, 3.0, 4.0],
                }
            ],
            "risk_count": 1,
            "risks": [
                {
                    "label": "knife",
                    "score": 0.91,
                    "box": [1.0, 2.0, 3.0, 4.0],
                    "risk_level": "high",
                }
            ],
            "result_image_path": "/tmp/result.jpg",
        }
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, image_path: Path, **kwargs) -> dict:
        call = {
            "image_path": Path(image_path),
            "exists_during_call": Path(image_path).exists(),
            "kwargs": kwargs,
        }
        self.calls.append(call)

        if self.error is not None:
            raise self.error

        return self.response


@pytest.fixture
def mocked_analysis_service(monkeypatch: pytest.MonkeyPatch) -> AnalyzeImageMock:
    mock = AnalyzeImageMock()
    module = types.ModuleType("app.services.image_analysis_service")
    module.analyze_image = mock
    monkeypatch.setitem(sys.modules, "app.services.image_analysis_service", module)
    return mock


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_accepts_valid_jpeg(
    client: TestClient,
    mocked_analysis_service: AnalyzeImageMock,
) -> None:
    response = client.post(
        "/analyze",
        files={
            "image": (
                "room.jpg",
                make_image_bytes("JPEG"),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["detection_count"] == 1
    assert response.json()["detections"][0]["label"] == "knife"
    assert len(mocked_analysis_service.calls) == 1

    call = mocked_analysis_service.calls[0]
    assert call["exists_during_call"] is True
    assert call["image_path"].exists() is False
    assert call["kwargs"]["output_directory"].name != "result.jpg"
    assert call["kwargs"]["result_image_path"].name == "result.jpg"
    assert call["kwargs"]["result_image_path"].parent == call["kwargs"]["output_directory"]


def test_analyze_accepts_valid_png(
    client: TestClient,
    mocked_analysis_service: AnalyzeImageMock,
) -> None:
    response = client.post(
        "/analyze",
        files={
            "image": (
                "room.png",
                make_image_bytes("PNG"),
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    assert response.json()["result_image_path"] == "/tmp/result.jpg"
    assert len(mocked_analysis_service.calls) == 1


def test_analyze_rejects_unsupported_content_type(
    client: TestClient,
    mocked_analysis_service: AnalyzeImageMock,
) -> None:
    response = client.post(
        "/analyze",
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "UNSUPPORTED_MEDIA_TYPE",
            "message": "Yalnızca JPEG ve PNG görseller desteklenmektedir.",
        }
    }
    assert mocked_analysis_service.calls == []


def test_analyze_rejects_corrupt_image(
    client: TestClient,
    mocked_analysis_service: AnalyzeImageMock,
) -> None:
    response = client.post(
        "/analyze",
        files={"image": ("broken.jpg", b"not a real jpeg", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "INVALID_IMAGE",
            "message": "Görsel içeriği okunamadı.",
        }
    }
    assert mocked_analysis_service.calls == []


def test_analyze_rejects_upload_over_size_limit(
    client: TestClient,
    mocked_analysis_service: AnalyzeImageMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api, "MAX_UPLOAD_SIZE_BYTES", 8)

    response = client.post(
        "/analyze",
        files={"image": ("large.jpg", b"x" * 9, "image/jpeg")},
    )

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "code": "FILE_TOO_LARGE",
            "message": "Dosya boyutu 10 MB sınırını aşıyor.",
        }
    }
    assert mocked_analysis_service.calls == []


def test_analyze_hides_unexpected_service_errors(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock = AnalyzeImageMock(error=RuntimeError("secret model failure"))
    module = types.ModuleType("app.services.image_analysis_service")
    module.analyze_image = mock
    monkeypatch.setitem(sys.modules, "app.services.image_analysis_service", module)

    with caplog.at_level(logging.ERROR, logger="app.main"):
        response = client.post(
            "/analyze",
            files={
                "image": (
                    "room.jpg",
                    make_image_bytes("JPEG"),
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "ANALYSIS_FAILED",
            "message": "Görsel analizi sırasında beklenmeyen bir hata oluştu.",
        }
    }
    assert "secret model failure" not in response.text
    assert len(mock.calls) == 1
    assert "analysis failed" in caplog.text
