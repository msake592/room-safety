from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from PIL import Image

from app.services.image_analysis_service import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_MIN_SCORE,
    DEFAULT_TEXT_PROMPT,
    DEFAULT_TEXT_THRESHOLD,
    IOU_THRESHOLD,
    analyze_image,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grounding DINO ile nesne tespiti yapar ve bulunan "
            "kutuları SAM2 ile maskeler."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        default="test-images/room.jpg",
        help="Repo köküne göre test görselinin yolu.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_TEXT_PROMPT,
        help="Grounding DINO metin prompt'u.",
    )

    parser.add_argument(
        "--box-threshold",
        type=float,
        default=DEFAULT_BOX_THRESHOLD,
        help="Grounding DINO box threshold değeri.",
    )

    parser.add_argument(
        "--text-threshold",
        type=float,
        default=DEFAULT_TEXT_THRESHOLD,
        help="Grounding DINO text threshold değeri.",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="SAM2 ve risk motoru öncesi minimum detection skoru.",
    )

    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=IOU_THRESHOLD,
        help="Aynı etikete ait kutular için NMS IoU threshold değeri.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Sonuç görsellerini matplotlib penceresinde gösterir.",
    )

    return parser.parse_args()


def show_result(image: Image.Image, title: str) -> None:
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def main() -> None:
    arguments = parse_arguments()

    result = analyze_image(
        arguments.image,
        prompt=arguments.prompt,
        box_threshold=arguments.box_threshold,
        text_threshold=arguments.text_threshold,
        min_score=arguments.min_score,
        nms_iou_threshold=arguments.nms_iou_threshold,
    )

    if arguments.show:
        show_result(
            Image.open(result["boxed_image_path"]),
            "Grounding DINO Tespitleri",
        )
        show_result(
            Image.open(result["result_image_path"]),
            "Grounding DINO + SAM2 Sonucu",
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest kullanıcı tarafından durduruldu.")
        raise SystemExit(130)
    except Exception as error:
        print(
            f"\nTest başarısız oldu: "
            f"{type(error).__name__}: {error}"
        )
        raise SystemExit(1)
