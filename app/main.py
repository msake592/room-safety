from __future__ import annotations

import logging
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.logging_config import configure_logging
from app.schemas.analysis import (
    AnalysisResponse,
    ApiError,
    ERROR_RESPONSES,
    ErrorCode,
    ErrorResponse,
    HealthResponse,
)


configure_logging()
logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_SIZE_MB = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
SUPPORTED_IMAGE_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}

app = FastAPI(title="Room Safety API")


@app.exception_handler(ApiError)
def handle_api_error(_, error: ApiError) -> JSONResponse:
    response = ErrorResponse(
        error={
            "code": error.code,
            "message": error.message,
        }
    )
    return JSONResponse(
        status_code=error.status_code,
        content=response.model_dump(mode="json"),
    )


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return {"status": "ok"}


def save_upload_to_temp_file(upload: UploadFile) -> Path:
    suffix = SUPPORTED_IMAGE_CONTENT_TYPES.get(upload.content_type)

    if suffix is None:
        raise ApiError(
            status_code=400,
            code=ErrorCode.UNSUPPORTED_MEDIA_TYPE,
            message="Yalnızca JPEG ve PNG görseller desteklenmektedir.",
        )

    temp_file = tempfile.NamedTemporaryFile(
        prefix="room-safety-upload-",
        suffix=suffix,
        delete=False,
    )
    temp_path = Path(temp_file.name)
    total_size = 0

    try:
        with temp_file:
            while True:
                chunk = upload.file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    raise ApiError(
                        status_code=413,
                        code=ErrorCode.FILE_TOO_LARGE,
                        message=(
                            f"Dosya boyutu {MAX_UPLOAD_SIZE_MB} MB "
                            "sınırını aşıyor."
                        ),
                    )

                temp_file.write(chunk)

        return temp_path
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def verify_image_file(image_path: Path) -> None:
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ApiError(
            status_code=400,
            code=ErrorCode.INVALID_IMAGE,
            message="Görsel içeriği okunamadı.",
        ) from error


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses=ERROR_RESPONSES,
)
def analyze(image: UploadFile = File(...)) -> dict:
    temp_image_path: Path | None = None
    request_id = uuid.uuid4().hex
    repo_root = get_repo_root()
    output_directory = repo_root / "outputs" / "api" / request_id
    result_image_path = output_directory / "result.jpg"
    request_start = time.perf_counter()
    uploaded_size = 0

    try:
        temp_image_path = save_upload_to_temp_file(image)
        uploaded_size = temp_image_path.stat().st_size
        logger.info(
            "analysis request accepted content_type=%s size_bytes=%s request_id=%s",
            image.content_type,
            uploaded_size,
            request_id,
        )
        verify_image_file(temp_image_path)

        from app.services.image_analysis_service import analyze_image

        result = analyze_image(
            temp_image_path,
            output_directory=output_directory,
            result_image_path=result_image_path,
        )

        elapsed = time.perf_counter() - request_start
        logger.info(
            "analysis completed request_id=%s elapsed_seconds=%.3f",
            request_id,
            elapsed,
        )

        return jsonable_encoder(result)
    except ApiError:
        raise
    except Exception as error:
        elapsed = time.perf_counter() - request_start
        logger.exception(
            "analysis failed request_id=%s elapsed_seconds=%.3f size_bytes=%s",
            request_id,
            elapsed,
            uploaded_size,
        )
        raise ApiError(
            status_code=500,
            code=ErrorCode.ANALYSIS_FAILED,
            message="Görsel analizi sırasında beklenmeyen bir hata oluştu.",
        ) from error
    finally:
        if temp_image_path is not None:
            temp_image_path.unlink(missing_ok=True)
        image.file.close()
