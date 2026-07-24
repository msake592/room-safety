from __future__ import annotations

from collections.abc import Sequence


DANGEROUS_OBJECT_LABELS: tuple[str, ...] = (
    "knife",
    "scissors",
    "medicine bottle",
    "pill bottle",
    "cleaning product bottle",
    "detergent bottle",
    "electrical outlet",
    "power outlet",
    "electrical cable",
    "power cable",
    "lighter",
    "matchbox",
    "glass bottle",
)


def build_grounding_dino_prompt(labels: Sequence[str] = DANGEROUS_OBJECT_LABELS) -> str:
    return ". ".join(labels) + "."
