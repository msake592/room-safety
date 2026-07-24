from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import pipeline

from app.config import DANGEROUS_OBJECT_LABELS


def resolve_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


image_path = Path("test-images/room.jpg")

if not image_path.exists():
    raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")

device = resolve_device()
print("Kullanılan cihaz:", device)

pipe = pipeline(
    task="zero-shot-object-detection",
    model="IDEA-Research/grounding-dino-tiny",
    device=device,
)

image = Image.open(image_path).convert("RGB")

results = pipe(
    image,
    candidate_labels=list(DANGEROUS_OBJECT_LABELS),
    threshold=0.40,
)

print("Bulunan nesne sayısı:", len(results))

for result in results:
    print(result)

annotated_image = image.copy()
draw = ImageDraw.Draw(annotated_image)

for result in results:
    box = result["box"]
    score = result["score"]
    label = result["label"]

    coordinates = (
        box["xmin"],
        box["ymin"],
        box["xmax"],
        box["ymax"],
    )

    draw.rectangle(coordinates, width=4)
    draw.text(
        (box["xmin"], max(0, box["ymin"] - 15)),
        f"{label}: {score:.2f}",
    )

annotated_image.save("result.jpg")
print("Sonuç result.jpg olarak kaydedildi.")
