from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from app.config import build_grounding_dino_prompt

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
IMAGE_PATH = Path("test-images/room.jpg")

# Önce MPS ihtimalini ortadan kaldırmak için CPU kullanıyoruz.
device = torch.device("cpu")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Görsel bulunamadı: {IMAGE_PATH.resolve()}"
    )

image = Image.open(IMAGE_PATH).convert("RGB")

print("Açılan görsel:", IMAGE_PATH.resolve())
print("Görsel boyutu:", image.size)
print("Çalışma cihazı:", device)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
model.to(device)
model.eval()

# Grounding DINO'da noktayla ayrılmış açıklama kullanmak daha güvenlidir.
text = build_grounding_dino_prompt()

inputs = processor(
    images=image,
    text=text,
    return_tensors="pt",
)

inputs = {
    key: value.to(device) if isinstance(value, torch.Tensor) else value
    for key, value in inputs.items()
}

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs["input_ids"],
    threshold=0.20,
    text_threshold=0.10,
    target_sizes=[image.size[::-1]],
)[0]

print("Bulunan kutu sayısı:", len(results["boxes"]))

draw = ImageDraw.Draw(image)

# Yeni Transformers sürümünde metin etiketleri burada bulunuyor.
text_labels = results.get("text_labels", results.get("labels", []))

for box, score, label in zip(
    results["boxes"],
    results["scores"],
    text_labels,
):
    x_min, y_min, x_max, y_max = box.tolist()
    score_value = score.item()

    print(
        f"Etiket: {label!r} | "
        f"Skor: {score_value:.4f} | "
        f"Kutu: {[round(value, 2) for value in box.tolist()]}"
    )

    # Çok düşük skorlu ve gereksiz kutuları çizme.
    if score_value < 0.20:
        continue

    draw.rectangle(
        [(x_min, y_min), (x_max, y_max)],
        outline="red",
        width=3,
    )

    draw.text(
        (x_min, max(0, y_min - 15)),
        f"{label} {score_value:.2f}",
        fill="red",
    )

output_path = Path("result.jpg")
image.save(output_path)

print("Sonuç görseli kaydedildi:", output_path.resolve())
