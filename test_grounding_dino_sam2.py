from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from app.risk_engine.engine import RiskEngine

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import transformers
from PIL import Image, ImageDraw
from torchvision.ops import nms
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)


GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"

DEFAULT_TEXT_PROMPT = "knife."
DEFAULT_BOX_THRESHOLD = 0.20
DEFAULT_TEXT_THRESHOLD = 0.10
DEFAULT_MIN_SCORE = 0.30
DEFAULT_NMS_IOU_THRESHOLD = 0.50
DEFAULT_MIN_BOX_AREA = 4.0


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
        default=DEFAULT_NMS_IOU_THRESHOLD,
        help="Aynı etikete ait kutular için NMS IoU threshold değeri.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Sonuç görsellerini matplotlib penceresinde gösterir.",
    )

    return parser.parse_args()


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_image_path(repo_root: Path, image_argument: str) -> Path:
    image_path = Path(image_argument).expanduser()

    if not image_path.is_absolute():
        image_path = repo_root / image_path

    image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Test görseli bulunamadı: {image_path}"
        )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Belirtilen yol bir dosya değil: {image_path}"
        )

    return image_path


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def move_tensors_to_device(
    values: dict,
    device: torch.device,
) -> dict:
    return {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in values.items()
    }


def print_environment_info(
    device: torch.device,
    image_path: Path,
    image: Image.Image,
    prompt: str,
    box_threshold: float,
    text_threshold: float,
    min_score: float,
    nms_iou_threshold: float,
) -> None:
    print("\n=== ORTAM BİLGİLERİ ===")
    print("Platform:", platform.platform())
    print("Python:", sys.version.split()[0])
    print("Pillow:", PIL.__version__)
    print("PyTorch:", torch.__version__)
    print("Transformers:", transformers.__version__)
    print("CUDA kullanılabilir:", torch.cuda.is_available())
    print("MPS kullanılabilir:", torch.backends.mps.is_available())
    print("Seçilen cihaz:", device)

    print("\n=== TEST AYARLARI ===")
    print("Grounding DINO modeli:", GROUNDING_MODEL_ID)
    print("SAM2 modeli:", SAM2_MODEL_ID)
    print("Görsel:", image_path)
    print("Görsel boyutu:", image.size)
    print("Görsel modu:", image.mode)
    print("Prompt:", prompt)
    print("Box threshold:", box_threshold)
    print("Text threshold:", text_threshold)
    print("Minimum detection skoru:", min_score)
    print("NMS IoU threshold:", nms_iou_threshold)


def load_grounding_dino(
    device: torch.device,
) -> tuple[AutoProcessor, AutoModelForZeroShotObjectDetection]:
    print("\nGrounding DINO yükleniyor...")

    processor = AutoProcessor.from_pretrained(
        GROUNDING_MODEL_ID
    )

    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_MODEL_ID
    )

    model.to(device)
    model.eval()

    print("Grounding DINO hazır.")

    return processor, model


def run_grounding_dino(
    image: Image.Image,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    device: torch.device,
) -> dict:
    inputs = processor(
        images=image,
        text=text_prompt,
        return_tensors="pt",
    )

    inputs = move_tensors_to_device(inputs, device)

    inference_start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]

    inference_duration = time.perf_counter() - inference_start

    print(
        f"Grounding DINO inference süresi: "
        f"{inference_duration:.2f} saniye"
    )

    return results


def extract_detections(
    grounding_results: dict,
    minimum_score: float,
    nms_iou_threshold: float,
    image_size: tuple[int, int],
) -> tuple[list[list[float]], list[float], list[str]]:
    boxes = grounding_results["boxes"]
    scores = grounding_results["scores"]

    text_labels = grounding_results.get("text_labels")

    if text_labels is None:
        raise RuntimeError(
            "Grounding DINO sonucu 'text_labels' alanını içermiyor. "
            "Transformers sürümünü kontrol edin."
        )

    print("\n=== TÜM GROUNDING DINO TESPİTLERİ ===")
    print("Bulunan kutu sayısı:", len(boxes))
    print("Ham tespit sayısı:", len(boxes))

    for index, (box, score, label) in enumerate(
        zip(boxes, scores, text_labels),
        start=1,
    ):
        box_values = box.detach().cpu().tolist()
        score_value = float(score.detach().cpu().item())

        x_min, y_min, x_max, y_max = box_values
        box_width = x_max - x_min
        box_height = y_max - y_min

        print(
            f"{index}. Etiket: {label} | "
            f"Skor: {score_value:.4f} | "
            f"Kutu: {[round(value, 2) for value in box_values]} | "
            f"Genişlik: {box_width:.2f} | "
            f"Yükseklik: {box_height:.2f}"
        )

    scored_boxes: list[list[float]] = []
    scored_scores: list[float] = []
    scored_labels: list[str] = []
    image_width, image_height = image_size

    for box, score, label in zip(
        boxes,
        scores,
        text_labels,
    ):
        score_value = float(score.detach().cpu().item())

        if score_value < minimum_score:
            continue

        x_min, y_min, x_max, y_max = [
            float(value)
            for value in box.detach().cpu().tolist()
        ]

        x_min = max(0.0, min(x_min, float(image_width)))
        y_min = max(0.0, min(y_min, float(image_height)))
        x_max = max(0.0, min(x_max, float(image_width)))
        y_max = max(0.0, min(y_max, float(image_height)))

        box_width = x_max - x_min
        box_height = y_max - y_min
        box_area = box_width * box_height

        if (
            x_max <= x_min
            or y_max <= y_min
            or box_area < DEFAULT_MIN_BOX_AREA
        ):
            continue

        scored_boxes.append([x_min, y_min, x_max, y_max])
        scored_scores.append(score_value)
        scored_labels.append(str(label))

    print("Skor filtresi sonrası tespit sayısı:", len(scored_boxes))

    if not scored_boxes:
        raise RuntimeError(
            "\nBelirlenen minimum skor üzerinde kutu bulunamadı.\n"
            "Olası sebepler:\n"
            "- Minimum skor veya threshold değerleri yüksek olabilir.\n"
            "- Prompt görseldeki nesne için uygun olmayabilir.\n"
            "- Grounding DINO nesneyi tespit edememiş olabilir."
        )

    selected_boxes: list[list[float]] = []
    selected_scores: list[float] = []
    selected_labels: list[str] = []

    for label in sorted(set(scored_labels)):
        label_indices = [
            index
            for index, current_label in enumerate(scored_labels)
            if current_label == label
        ]
        label_boxes = torch.tensor(
            [scored_boxes[index] for index in label_indices],
            dtype=torch.float32,
        )
        label_scores = torch.tensor(
            [scored_scores[index] for index in label_indices],
            dtype=torch.float32,
        )
        kept_local_indices = nms(
            label_boxes,
            label_scores,
            nms_iou_threshold,
        ).tolist()

        for kept_local_index in kept_local_indices:
            original_index = label_indices[kept_local_index]
            selected_boxes.append(scored_boxes[original_index])
            selected_scores.append(scored_scores[original_index])
            selected_labels.append(scored_labels[original_index])

    ordered_indices = sorted(
        range(len(selected_scores)),
        key=lambda index: selected_scores[index],
        reverse=True,
    )
    selected_boxes = [selected_boxes[index] for index in ordered_indices]
    selected_scores = [selected_scores[index] for index in ordered_indices]
    selected_labels = [selected_labels[index] for index in ordered_indices]

    print("NMS sonrası tespit sayısı:", len(selected_boxes))

    print("\n=== SAM2 VE RİSK MOTORU İÇİN FİLTRELENMİŞ TESPİTLER ===")

    for index, (box, score, label) in enumerate(
        zip(
            selected_boxes,
            selected_scores,
            selected_labels,
        ),
        start=1,
    ):
        print(
            f"{index}. Etiket: {label} | "
            f"Skor: {score:.4f} | "
            f"Kutu: {[round(value, 2) for value in box]}"
        )
        print(
            f"label={label} "
            f"score={score:.4f} "
            f"box={[round(value, 2) for value in box]}"
        )

    return selected_boxes, selected_scores, selected_labels


def draw_detection_boxes(
    image: Image.Image,
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
    output_path: Path,
) -> Image.Image:
    boxed_image = image.copy()
    draw = ImageDraw.Draw(boxed_image)

    for index, (box, score, label) in enumerate(
        zip(boxes, scores, labels),
        start=1,
    ):
        x_min, y_min, x_max, y_max = box

        draw.rectangle(
            (x_min, y_min, x_max, y_max),
            outline="red",
            width=4,
        )

        draw.text(
            (x_min, max(0, y_min - 18)),
            f"{index}. {label} {score:.2f}",
            fill="red",
        )

    boxed_image.save(
        output_path,
        format="JPEG",
        quality=95,
    )

    print("Kutu sonucu kaydedildi:", output_path)

    return boxed_image


def build_risk_detections(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
) -> list[dict]:
    if not (
        len(boxes)
        == len(scores)
        == len(labels)
    ):
        raise ValueError(
            "Bounding box, score ve label sayıları birbiriyle uyuşmuyor."
        )

    detections = []

    for box, score, label in zip(boxes, scores, labels):
        detections.append(
            {
                "label": str(label),
                "score": float(score),
                "box": [float(value) for value in box],
            }
        )

    return detections


def print_risk_summary(
    detections: list[dict],
    risk_results: list[dict],
) -> None:
    print("\n=== RİSK ANALİZİ ===")
    print("Grounding DINO detection sayısı:", len(detections))
    print("Risk sonucu sayısı:", len(risk_results))

    for index, risk in enumerate(risk_results, start=1):
        print(f"\nRisk {index}")
        print("Label:", risk.get("label"))
        print(
            "Detection score:",
            f"{float(risk.get('score', 0.0)):.2f}",
        )
        print("Risk level:", risk.get("risk_level"))
        print("Risk score:", risk.get("risk_score"))
        print("Reason:", risk.get("reason"))
        print("Recommendation:", risk.get("recommendation"))
        print("Box:", risk.get("box"))


def save_risk_analysis(
    image_path: Path,
    risk_engine: RiskEngine,
    detections: list[dict],
    risk_results: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "image_path": str(image_path),
        "target_group": risk_engine.target_group,
        "detection_count": len(detections),
        "risk_count": len(risk_results),
        "detections": detections,
        "risks": risk_results,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Risk analizi kaydedildi:", output_path)


def load_sam2(
    device: torch.device,
) -> tuple[Sam2Processor, Sam2Model]:
    print("\nSAM2 yükleniyor...")

    processor = Sam2Processor.from_pretrained(
        SAM2_MODEL_ID
    )

    model = Sam2Model.from_pretrained(
        SAM2_MODEL_ID
    )

    model.to(device)
    model.eval()

    print("SAM2 hazır.")

    return processor, model


def run_sam2(
    image: Image.Image,
    boxes: list[list[float]],
    processor: Sam2Processor,
    model: Sam2Model,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    input_boxes = [boxes]

    sam_inputs = processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt",
    )

    sam_inputs = move_tensors_to_device(
        sam_inputs,
        device,
    )

    inference_start = time.perf_counter()

    with torch.inference_mode():
        sam_outputs = model(
            **sam_inputs,
            multimask_output=False,
        )

    inference_duration = time.perf_counter() - inference_start

    masks = processor.post_process_masks(
        sam_outputs.pred_masks.detach().cpu(),
        sam_inputs["original_sizes"].detach().cpu(),
    )[0]

    iou_scores = sam_outputs.iou_scores.detach().cpu()

    print(
        f"SAM2 inference süresi: "
        f"{inference_duration:.2f} saniye"
    )
    print("Kutu sayısı:", len(boxes))
    print("Maske tensor boyutu:", tuple(masks.shape))
    print("IoU skor tensor boyutu:", tuple(iou_scores.shape))
    print("IoU skorları:", iou_scores)

    return masks, iou_scores, inference_duration


def create_segmentation_result(
    image: Image.Image,
    masks: torch.Tensor,
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
    output_path: Path,
) -> Image.Image:
    image_array = np.asarray(image).copy()
    overlay = image_array.copy()

    mask_colors = [
        np.array([255, 0, 0], dtype=np.float32),
        np.array([0, 255, 0], dtype=np.float32),
        np.array([0, 0, 255], dtype=np.float32),
        np.array([255, 255, 0], dtype=np.float32),
        np.array([255, 0, 255], dtype=np.float32),
        np.array([0, 255, 255], dtype=np.float32),
    ]

    for index in range(masks.shape[0]):
        mask_tensor = masks[index]

        if mask_tensor.ndim == 3:
            mask_tensor = mask_tensor[0]

        mask = mask_tensor.numpy().astype(bool)

        color = mask_colors[index % len(mask_colors)]

        overlay[mask] = (
            0.45 * overlay[mask].astype(np.float32)
            + 0.55 * color
        ).astype(np.uint8)

    result_image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(result_image)

    for index, (box, score, label) in enumerate(
        zip(boxes, scores, labels),
        start=1,
    ):
        x_min, y_min, x_max, y_max = box

        draw.rectangle(
            (x_min, y_min, x_max, y_max),
            outline="red",
            width=3,
        )

        draw.text(
            (x_min, max(0, y_min - 18)),
            f"{index}. {label} | DINO: {score:.2f}",
            fill="red",
        )

    print("result.jpg kaydediliyor...")
    result_image.save(
        output_path,
        format="JPEG",
        quality=95,
    )

    print(f"result.jpg kaydedildi: {output_path.resolve()}")
    print("Maskeleme sonucu kaydedildi:", output_path)

    return result_image


def show_result(
    image: Image.Image,
    title: str,
) -> None:
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()


def main() -> None:
    arguments = parse_arguments()

    total_start = time.perf_counter()

    repo_root = get_repo_root()
    output_directory = repo_root / "outputs"
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_path = resolve_image_path(
        repo_root,
        arguments.image,
    )

    image = Image.open(image_path).convert("RGB")
    device = select_device()

    print_environment_info(
        device=device,
        image_path=image_path,
        image=image,
        prompt=arguments.prompt,
        box_threshold=arguments.box_threshold,
        text_threshold=arguments.text_threshold,
        min_score=arguments.min_score,
        nms_iou_threshold=arguments.nms_iou_threshold,
    )

    grounding_processor, grounding_model = load_grounding_dino(
        device=device,
    )

    grounding_results = run_grounding_dino(
        image=image,
        text_prompt=arguments.prompt,
        box_threshold=arguments.box_threshold,
        text_threshold=arguments.text_threshold,
        processor=grounding_processor,
        model=grounding_model,
        device=device,
    )

    boxes, scores, labels = extract_detections(
        grounding_results=grounding_results,
        minimum_score=arguments.min_score,
        nms_iou_threshold=arguments.nms_iou_threshold,
        image_size=image.size,
    )

    detections = build_risk_detections(
        boxes=boxes,
        scores=scores,
        labels=labels,
    )

    risk_engine = RiskEngine()
    risk_results = risk_engine.evaluate(detections)
    print("Üretilen risk sayısı:", len(risk_results))

    print_risk_summary(
        detections=detections,
        risk_results=risk_results,
    )

    risk_output_path = output_directory / "risk_analysis.json"
    save_risk_analysis(
        image_path=image_path,
        risk_engine=risk_engine,
        detections=detections,
        risk_results=risk_results,
        output_path=risk_output_path,
    )

    boxed_output_path = (
        output_directory
        / "grounding_dino_local_result.jpg"
    )

    boxed_image = draw_detection_boxes(
        image=image,
        boxes=boxes,
        scores=scores,
        labels=labels,
        output_path=boxed_output_path,
    )

    sam_processor, sam_model = load_sam2(
        device=device,
    )

    masks, _, _ = run_sam2(
        image=image,
        boxes=boxes,
        processor=sam_processor,
        model=sam_model,
        device=device,
    )

    if masks.shape[0] != len(detections):
        raise ValueError(
            "SAM2 maske sayısı ile detection sayısı birbiriyle uyuşmuyor."
        )

    segmented_output_path = repo_root / "result.jpg"

    segmented_image = create_segmentation_result(
        image=image,
        masks=masks,
        boxes=boxes,
        scores=scores,
        labels=labels,
        output_path=segmented_output_path,
    )

    if arguments.show:
        show_result(
            boxed_image,
            "Grounding DINO Tespitleri",
        )

        show_result(
            segmented_image,
            "Grounding DINO + SAM2 Sonucu",
        )

    total_duration = time.perf_counter() - total_start

    print("\n=== TEST TAMAMLANDI ===")
    print("Toplam seçilen nesne sayısı:", len(boxes))
    print("Risk analizi:", risk_output_path)
    print("Kutu sonucu:", boxed_output_path)
    print("Maskeleme sonucu:", segmented_output_path)
    print(f"Toplam çalışma süresi: {total_duration:.2f} saniye")


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
