from __future__ import annotations

from pathlib import Path

import pytest

from app.risk_engine.engine import RiskEngine
from app.risk_engine.labels import describe_hazard_label, normalize_hazard_label


def make_detection(label: str) -> dict:
    return {
        "label": label,
        "raw_label": label,
        "score": 0.9,
        "box": [1.0, 2.0, 3.0, 4.0],
    }


@pytest.fixture
def risk_engine() -> RiskEngine:
    return RiskEngine()


@pytest.mark.parametrize(
    ("raw_label", "expected_canonical_label", "expected_rule_id"),
    [
        ("scissors", "scissors", "sharp_object_scissors"),
        (
            "electrical outlet power outlet",
            "electrical_outlet",
            "electrical_outlet_access",
        ),
        (
            "electrical outlet power outlet electrical cable power cable",
            "electrical_outlet",
            "electrical_outlet_access",
        ),
        ("medicine bottle pill bottle", "medicine_bottle", "medicine_access"),
        (
            "cleaning product bottle detergent bottle",
            "detergent_bottle",
            "chemical_product_access",
        ),
        ("kitchen knife", "knife", "sharp_object_knife"),
    ],
)
def test_normalized_hazard_labels_match_expected_rules(
    risk_engine: RiskEngine,
    raw_label: str,
    expected_canonical_label: str,
    expected_rule_id: str,
) -> None:
    risks = risk_engine.evaluate([make_detection(raw_label)])

    assert normalize_hazard_label(raw_label) == expected_canonical_label
    assert len(risks) == 1
    assert risks[0]["rule_id"] == expected_rule_id
    assert risks[0]["label"] == raw_label
    assert risks[0]["raw_label"] == raw_label
    assert risks[0]["canonical_label"] == expected_canonical_label
    assert risks[0]["display_label"]


def test_combined_medicine_and_detergent_label_does_not_default_to_medicine(
    risk_engine: RiskEngine,
) -> None:
    raw_label = "medicine bottle bottle detergent bottle glass bottle"

    risks = risk_engine.evaluate([make_detection(raw_label)])
    normalized_label = describe_hazard_label(raw_label)

    assert normalized_label is not None
    assert normalized_label.canonical_label == "detergent_bottle"
    assert normalized_label.display_label == "Detergent bottle"
    assert len(risks) == 1
    assert risks[0]["rule_id"] == "chemical_product_access"
    assert risks[0]["canonical_label"] == "detergent_bottle"
    assert risks[0]["display_label"] == "Detergent bottle"


@pytest.mark.parametrize(
    "raw_label",
    [
        "remote control",
        "glass bottle",
        "water bottle",
    ],
)
def test_unrelated_or_ambiguous_bottle_labels_do_not_match_hazard_rules(
    risk_engine: RiskEngine,
    raw_label: str,
) -> None:
    risks = risk_engine.evaluate([make_detection(raw_label)])

    assert normalize_hazard_label(raw_label) == raw_label
    assert risks == []


def test_frontend_hazard_card_uses_display_label_for_title() -> None:
    component_path = (
        Path(__file__).resolve().parents[1]
        / "frontend"
        / "src"
        / "components"
        / "AnalysisResult.tsx"
    )
    component_source = component_path.read_text(encoding="utf-8")

    assert "const title = risk.display_label ?? risk.canonical_label ?? risk.label" in component_source
    assert '<span className="hazard-label">{title}</span>' in component_source
    assert "Raw: {detection.raw_label}" in component_source
