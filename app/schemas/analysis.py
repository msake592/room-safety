from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ErrorCode(str, Enum):
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    INVALID_IMAGE = "INVALID_IMAGE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str


class DetectionResponse(BaseModel):
    label: str
    score: float
    box: list[float]


class RiskResponse(DetectionResponse):
    model_config = ConfigDict(extra="allow")

    rule_id: str | None = None
    risk_level: str | None = None
    risk_score: int | float | None = None
    reason: str | None = None
    recommendation: str | None = None
    target_group: str | None = None


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    image_name: str
    image_path: str | None = None
    detection_count: int
    detections: list[DetectionResponse]
    risk_count: int | None = None
    risks: list[RiskResponse]
    result_image_path: str
    risk_analysis_path: str | None = None
    boxed_image_path: str | None = None


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: ErrorCode,
        message: str,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


ERROR_RESPONSES: dict[int, dict[str, Any]] = {
    400: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}
