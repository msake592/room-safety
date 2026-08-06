from __future__ import annotations

from dataclasses import dataclass
import re


_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizedHazardLabel:
    raw_label: str
    canonical_label: str
    display_label: str


@dataclass(frozen=True)
class HazardCategory:
    canonical_label: str
    display_label: str
    phrases: tuple[str, ...]


_HAZARD_CATEGORIES: tuple[HazardCategory, ...] = (
    HazardCategory(
        canonical_label="knife",
        display_label="Knife",
        phrases=(
            "kitchen knife",
            "bread knife",
            "knife",
            "cleaver",
        ),
    ),
    HazardCategory(
        canonical_label="scissors",
        display_label="Scissors",
        phrases=(
            "pair of scissors",
            "scissors",
        ),
    ),
    HazardCategory(
        canonical_label="electrical_outlet",
        display_label="Electrical outlet",
        phrases=(
            "electrical outlet",
            "power outlet",
        ),
    ),
    HazardCategory(
        canonical_label="medicine_bottle",
        display_label="Medicine bottle",
        phrases=(
            "medicine bottle",
            "pill bottle",
            "pill container",
            "medicine",
        ),
    ),
    HazardCategory(
        canonical_label="detergent_bottle",
        display_label="Detergent bottle",
        phrases=(
            "cleaning product bottle",
            "detergent bottle",
            "chemical product bottle",
            "cleaning product",
            "chemical product",
            "detergent",
            "chemical",
        ),
    ),
)


def clean_label(raw_label: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", raw_label.lower().strip()).rstrip(".")


def _matching_phrases(label: str, category: HazardCategory) -> tuple[str, ...]:
    return tuple(phrase for phrase in category.phrases if phrase in label)


def _select_best_category(label: str) -> HazardCategory | None:
    matches = [
        (category, _matching_phrases(label, category))
        for category in _HAZARD_CATEGORIES
    ]
    matches = [
        (category, phrases)
        for category, phrases in matches
        if phrases
    ]

    if not matches:
        return None

    if len(matches) == 1:
        return matches[0][0]

    category_by_label = {
        category.canonical_label: (category, phrases)
        for category, phrases in matches
    }

    # Grounding DINO can concatenate prompt phrases for one object. Without
    # per-phrase confidence, strong class-specific terms break ambiguity first.
    if (
        "medicine_bottle" in category_by_label
        and "pill bottle" in category_by_label["medicine_bottle"][1]
    ):
        return category_by_label["medicine_bottle"][0]

    if "detergent_bottle" in category_by_label and (
        "detergent bottle" in category_by_label["detergent_bottle"][1]
        or "cleaning product bottle" in category_by_label["detergent_bottle"][1]
    ):
        return category_by_label["detergent_bottle"][0]

    return sorted(
        matches,
        key=lambda item: (
            len(item[1]),
            max(len(phrase) for phrase in item[1]),
        ),
        reverse=True,
    )[0][0]


def describe_hazard_label(raw_label: str) -> NormalizedHazardLabel | None:
    if not isinstance(raw_label, str):
        return None

    cleaned_label = clean_label(raw_label)

    if not cleaned_label:
        return None

    category = _select_best_category(cleaned_label)

    if category is None:
        return NormalizedHazardLabel(
            raw_label=str(raw_label),
            canonical_label=cleaned_label,
            display_label=cleaned_label,
        )

    return NormalizedHazardLabel(
        raw_label=str(raw_label),
        canonical_label=category.canonical_label,
        display_label=category.display_label,
    )


def normalize_hazard_label(raw_label: str) -> str | None:
    normalized_label = describe_hazard_label(raw_label)

    if normalized_label is None:
        return None

    return normalized_label.canonical_label
