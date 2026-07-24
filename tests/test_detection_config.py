from __future__ import annotations

from pathlib import Path

from app.config import DANGEROUS_OBJECT_LABELS, build_grounding_dino_prompt


def test_dangerous_object_labels_include_mvp_labels() -> None:
    labels = set(DANGEROUS_OBJECT_LABELS)

    assert "knife" in labels
    assert "scissors" in labels
    assert "electrical outlet" in labels
    assert "lighter" in labels


def test_grounding_dino_prompt_is_built_from_config_labels() -> None:
    prompt = build_grounding_dino_prompt()

    for label in DANGEROUS_OBJECT_LABELS:
        assert f"{label}." in prompt


def test_detection_service_uses_configured_grounding_dino_prompt() -> None:
    service_path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "services"
        / "image_analysis_service.py"
    )
    service_source = service_path.read_text(encoding="utf-8")

    assert "from app.config import build_grounding_dino_prompt" in service_source
    assert "DEFAULT_TEXT_PROMPT = build_grounding_dino_prompt()" in service_source
    assert 'DEFAULT_TEXT_PROMPT = "knife."' not in service_source
