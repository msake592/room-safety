from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from PIL import Image, UnidentifiedImageError


logger = logging.getLogger(__name__)

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}

app = FastAPI(title="Room Safety API")


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def save_upload_to_temp_file(upload: UploadFile) -> Path:
    suffix = SUPPORTED_IMAGE_CONTENT_TYPES.get(upload.content_type)

    if suffix is None:
        raise HTTPException(
            status_code=400,
            detail="Desteklenmeyen dosya türü. JPEG veya PNG yükleyin.",
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
                    raise HTTPException(
                        status_code=413,
                        detail="Dosya boyutu 10 MB sınırını aşıyor.",
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
        raise HTTPException(
            status_code=400,
            detail="Görsel içeriği okunamadı.",
        ) from error


@app.post("/analyze")
def analyze(image: UploadFile = File(...)) -> dict:
    temp_image_path: Path | None = None
    request_id = uuid.uuid4().hex
    repo_root = get_repo_root()
    output_directory = repo_root / "outputs" / "api" / request_id
    result_image_path = output_directory / "result.jpg"

    try:
        temp_image_path = save_upload_to_temp_file(image)
        verify_image_file(temp_image_path)

        from app.services.image_analysis_service import analyze_image

        result = analyze_image(
            temp_image_path,
            output_directory=output_directory,
            result_image_path=result_image_path,
        )

        return jsonable_encoder(result)
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Image analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Görsel analizi sırasında beklenmeyen bir hata oluştu.",
        ) from error
    finally:
        if temp_image_path is not None:
            temp_image_path.unlink(missing_ok=True)
        image.file.close()
