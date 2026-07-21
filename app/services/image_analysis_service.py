from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import PIL
import torch
import transformers
from PIL import Image, ImageDraw, UnidentifiedImageError
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam2Model,
    Sam2Processor,
)

from app.risk_engine.engine import RiskEngine


GROUNDING_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_MODEL_ID = "facebook/sam2.1-hiera-tiny"

DEFAULT_TEXT_PROMPT = "knife."
DEFAULT_BOX_THRESHOLD = 0.20
DEFAULT_TEXT_THRESHOLD = 0.10
DEFAULT_MIN_SCORE = 0.30
IOU_THRESHOLD = 0.50
DEFAULT_MIN_BOX_AREA = 4.0

_DEVICE: torch.device | None = None
_GROUNDING_PROCESSOR: AutoProcessor | None = None
_GROUNDING_MODEL: AutoModelForZeroShotObjectDetection | None = None
_SAM_PROCESSOR: Sam2Processor | None = None
_SAM_MODEL: Sam2Model | None = None
_RISK_ENGINE: RiskEngine | None = None


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_image_path(repo_root: Path, image_argument: str | Path) -> Path:
    image_path = Path(image_argument).expanduser()

    if not image_path.is_absolute():
        image_path = repo_root / image_path

    image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Test görseli bulunamadı: {image_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Belirtilen yol bir dosya değil: {image_path}")

    return image_path


def load_image(image_path: Path) -> Image.Image:
    try:
        return Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as error:
        raise ValueError(f"Görsel açılamadı veya desteklenmiyor: {image_path}") from error
    except OSError as error:
        raise OSError(f"Görsel okunamadı: {image_path}") from error


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_device() -> torch.device:
    global _DEVICE

    if _DEVICE is None:
        _DEVICE = select_device()

    return _DEVICE


def move_tensors_to_device(values: dict, device: torch.device) -> dict:
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

    processor = AutoProcessor.from_pretrained(GROUNDING_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL_ID)

    model.to(device)
    model.eval()

    print("Grounding DINO hazır.")

    return processor, model


def get_grounding_dino(
    device: torch.device,
) -> tuple[AutoProcessor, AutoModelForZeroShotObjectDetection]:
    global _GROUNDING_MODEL, _GROUNDING_PROCESSOR

    if _GROUNDING_PROCESSOR is None or _GROUNDING_MODEL is None:
        _GROUNDING_PROCESSOR, _GROUNDING_MODEL = load_grounding_dino(device)

    return _GROUNDING_PROCESSOR, _GROUNDING_MODEL


def run_grounding_dino(
    image: Image.Image,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    processor: AutoProcessor,
    model: AutoModelForZeroShotObjectDetection,
    device: torch.device,
) -> dict:
    inputs = processor(images=image, text=text_prompt, return_tensors="pt")
    inputs = move_tensors_to_device(inputs, device)

    inference_start = time.perf_counter()

    try:
        with torch.inference_mode():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
    except Exception as error:
        raise RuntimeError("Grounding DINO inference başarısız oldu.") from error

    inference_duration = time.perf_counter() - inference_start
    print(f"Grounding DINO inference süresi: {inference_duration:.2f} saniye")

    return results


def normalize_label(label: str) -> str:
    return " ".join(str(label).lower().strip().split())


def calculate_iou(first_box: list[float], second_box: list[float]) -> float:
    first_x_min, first_y_min, first_x_max, first_y_max = first_box
    second_x_min, second_y_min, second_x_max, second_y_max = second_box

    intersection_x_min = max(first_x_min, second_x_min)
    intersection_y_min = max(first_y_min, second_y_min)
    intersection_x_max = min(first_x_max, second_x_max)
    intersection_y_max = min(first_y_max, second_y_max)

    intersection_width = max(0.0, intersection_x_max - intersection_x_min)
    intersection_height = max(0.0, intersection_y_max - intersection_y_min)
    intersection_area = intersection_width * intersection_height

    first_area = max(0.0, first_x_max - first_x_min) * max(
        0.0,
        first_y_max - first_y_min,
    )
    second_area = max(0.0, second_x_max - second_x_min) * max(
        0.0,
        second_y_max - second_y_min,
    )
    union_area = first_area + second_area - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def apply_class_aware_nms(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
    iou_threshold: float,
) -> tuple[list[list[float]], list[float], list[str]]:
    remaining_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )
    kept_indices: list[int] = []

    while remaining_indices:
        current_index = remaining_indices.pop(0)
        kept_indices.append(current_index)
        current_label = normalize_label(labels[current_index])

        next_remaining_indices = []

        for candidate_index in remaining_indices:
            candidate_label = normalize_label(labels[candidate_index])

            if candidate_label != current_label:
                next_remaining_indices.append(candidate_index)
                continue

            iou = calculate_iou(boxes[current_index], boxes[candidate_index])

            if iou <= iou_threshold:
                next_remaining_indices.append(candidate_index)

        remaining_indices = next_remaining_indices

    return (
        [boxes[index] for index in kept_indices],
        [scores[index] for index in kept_indices],
        [labels[index] for index in kept_indices],
    )


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

    for box, score, label in zip(boxes, scores, text_labels):
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

        if x_max <= x_min or y_max <= y_min or box_area < DEFAULT_MIN_BOX_AREA:
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

    selected_boxes, selected_scores, selected_labels = apply_class_aware_nms(
        boxes=scored_boxes,
        scores=scored_scores,
        labels=scored_labels,
        iou_threshold=nms_iou_threshold,
    )

    print("NMS sonrası tespit sayısı:", len(selected_boxes))
    print("\n=== SAM2 VE RİSK MOTORU İÇİN FİLTRELENMİŞ TESPİTLER ===")

    for index, (box, score, label) in enumerate(
        zip(selected_boxes, selected_scores, selected_labels),
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
        draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=4)
        draw.text(
            (x_min, max(0, y_min - 18)),
            f"{index}. {label} {score:.2f}",
            fill="red",
        )

    boxed_image.save(output_path, format="JPEG", quality=95)
    print("Kutu sonucu kaydedildi:", output_path)

    return boxed_image


def build_risk_detections(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
) -> list[dict]:
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("Bounding box, score ve label sayıları birbiriyle uyuşmuyor.")

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


def get_risk_engine() -> RiskEngine:
    global _RISK_ENGINE

    if _RISK_ENGINE is None:
        _RISK_ENGINE = RiskEngine()

    return _RISK_ENGINE


def print_risk_summary(detections: list[dict], risk_results: list[dict]) -> None:
    print("\n=== RİSK ANALİZİ ===")
    print("Grounding DINO detection sayısı:", len(detections))
    print("Risk sonucu sayısı:", len(risk_results))

    for index, risk in enumerate(risk_results, start=1):
        print(f"\nRisk {index}")
        print("Label:", risk.get("label"))
        print("Detection score:", f"{float(risk.get('score', 0.0)):.2f}")
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
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("Risk analizi kaydedildi:", output_path)


def load_sam2(device: torch.device) -> tuple[Sam2Processor, Sam2Model]:
    print("\nSAM2 yükleniyor...")

    processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
    model = Sam2Model.from_pretrained(SAM2_MODEL_ID)

    model.to(device)
    model.eval()

    print("SAM2 hazır.")

    return processor, model


def get_sam2(device: torch.device) -> tuple[Sam2Processor, Sam2Model]:
    global _SAM_MODEL, _SAM_PROCESSOR

    if _SAM_PROCESSOR is None or _SAM_MODEL is None:
        _SAM_PROCESSOR, _SAM_MODEL = load_sam2(device)

    return _SAM_PROCESSOR, _SAM_MODEL


def run_sam2(
    image: Image.Image,
    boxes: list[list[float]],
    processor: Sam2Processor,
    model: Sam2Model,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    input_boxes = [boxes]
    sam_inputs = processor(images=image, input_boxes=input_boxes, return_tensors="pt")
    sam_inputs = move_tensors_to_device(sam_inputs, device)
    inference_start = time.perf_counter()

    try:
        with torch.inference_mode():
            sam_outputs = model(**sam_inputs, multimask_output=False)

        masks = processor.post_process_masks(
            sam_outputs.pred_masks.detach().cpu(),
            sam_inputs["original_sizes"].detach().cpu(),
        )[0]
        iou_scores = sam_outputs.iou_scores.detach().cpu()
    except Exception as error:
        raise RuntimeError("SAM2 inference başarısız oldu.") from error

    inference_duration = time.perf_counter() - inference_start

    print(f"SAM2 inference süresi: {inference_duration:.2f} saniye")
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
        draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=3)
        draw.text(
            (x_min, max(0, y_min - 18)),
            f"{index}. {label} | DINO: {score:.2f}",
            fill="red",
        )

    print("result.jpg kaydediliyor...")
    result_image.save(output_path, format="JPEG", quality=95)
    print(f"result.jpg kaydedildi: {output_path.resolve()}")
    print("Maskeleme sonucu kaydedildi:", output_path)

    return result_image


def analyze_image(
    image_path: str | Path,
    *,
    prompt: str = DEFAULT_TEXT_PROMPT,
    box_threshold: float = DEFAULT_BOX_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    min_score: float = DEFAULT_MIN_SCORE,
    nms_iou_threshold: float = IOU_THRESHOLD,
    output_directory: str | Path | None = None,
    result_image_path: str | Path | None = None,
    print_debug: bool = True,
) -> dict:
    total_start = time.perf_counter()
    repo_root = get_repo_root()

    resolved_image_path = resolve_image_path(repo_root, image_path)
    image = load_image(resolved_image_path)
    device = get_device()

    if print_debug:
        print_environment_info(
            device=device,
            image_path=resolved_image_path,
            image=image,
            prompt=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            min_score=min_score,
            nms_iou_threshold=nms_iou_threshold,
        )

    output_path = Path(output_directory) if output_directory is not None else repo_root / "outputs"
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    segmented_output_path = (
        Path(result_image_path)
        if result_image_path is not None
        else repo_root / "result.jpg"
    )
    if not segmented_output_path.is_absolute():
        segmented_output_path = repo_root / segmented_output_path

    grounding_processor, grounding_model = get_grounding_dino(device)
    grounding_results = run_grounding_dino(
        image=image,
        text_prompt=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        processor=grounding_processor,
        model=grounding_model,
        device=device,
    )

    boxes, scores, labels = extract_detections(
        grounding_results=grounding_results,
        minimum_score=min_score,
        nms_iou_threshold=nms_iou_threshold,
        image_size=image.size,
    )

    detections = build_risk_detections(boxes=boxes, scores=scores, labels=labels)

    risk_engine = get_risk_engine()
    risk_results = risk_engine.evaluate(detections)
    print("Üretilen risk sayısı:", len(risk_results))
    print_risk_summary(detections=detections, risk_results=risk_results)

    risk_output_path = output_path / "risk_analysis.json"
    save_risk_analysis(
        image_path=resolved_image_path,
        risk_engine=risk_engine,
        detections=detections,
        risk_results=risk_results,
        output_path=risk_output_path,
    )

    boxed_output_path = output_path / "grounding_dino_local_result.jpg"
    draw_detection_boxes(
        image=image,
        boxes=boxes,
        scores=scores,
        labels=labels,
        output_path=boxed_output_path,
    )

    sam_processor, sam_model = get_sam2(device)
    masks, _, _ = run_sam2(
        image=image,
        boxes=boxes,
        processor=sam_processor,
        model=sam_model,
        device=device,
    )

    if masks.shape[0] != len(detections):
        raise ValueError("SAM2 maske sayısı ile detection sayısı birbiriyle uyuşmuyor.")

    create_segmentation_result(
        image=image,
        masks=masks,
        boxes=boxes,
        scores=scores,
        labels=labels,
        output_path=segmented_output_path,
    )

    total_duration = time.perf_counter() - total_start

    print("\n=== TEST TAMAMLANDI ===")
    print("Toplam seçilen nesne sayısı:", len(boxes))
    print("Risk analizi:", risk_output_path)
    print("Kutu sonucu:", boxed_output_path)
    print("Maskeleme sonucu:", segmented_output_path)
    print(f"Toplam çalışma süresi: {total_duration:.2f} saniye")

    return {
        "image_name": resolved_image_path.name,
        "image_path": str(resolved_image_path),
        "detection_count": len(detections),
        "detections": detections,
        "risk_count": len(risk_results),
        "risks": risk_results,
        "result_image_path": str(segmented_output_path),
        "risk_analysis_path": str(risk_output_path),
        "boxed_image_path": str(boxed_output_path),
    }
