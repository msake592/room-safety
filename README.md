# Room Safety

Room Safety is a computer vision project that analyzes indoor images and identifies objects that may pose a safety risk for young children. It combines object detection, image segmentation, and a rule-based risk engine to generate simple safety assessments from a single image.

The current version focuses on children between the ages of 2 and 6 and demonstrates how AI models can be integrated into a lightweight API service.

---

## Features

- Detects potentially dangerous objects in indoor environments
- Segments detected objects for more accurate localization
- Evaluates detected objects using a rule-based risk engine
- Returns risk level, explanation, and safety recommendation
- REST API built with FastAPI
- Docker support for containerized deployment

---

## How It Works

The analysis pipeline follows these steps:

1. An image is uploaded to the API.
2. Grounding DINO detects relevant objects.
3. SAM2 generates segmentation masks for detected objects.
4. The detected objects are passed to the risk engine.
5. The risk engine evaluates each object according to predefined safety rules.
6. A structured JSON response is returned with detected risks and recommendations.

---

## Tech Stack

### Backend

- FastAPI
- Python

### Computer Vision

- Grounding DINO
- SAM2
- PyTorch
- Hugging Face Transformers
- OpenCV

---

## Project Structure

```text
room-safety/
│
├── app/
│   ├── api/
│   ├── models/
│   ├── risk_engine/
│   ├── services/
│   └── main.py
│
├── config/
├── test-images/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## API

### Health Check

```http
GET /health
```

Returns the current API status.

### Analyze Image

```http
POST /analyze
```

Uploads an image and returns detected objects together with their risk evaluation.

Example response:

```json
{
  "detections": [
    {
      "label": "kitchen knife",
      "risk_level": "high",
      "risk_score": 90,
      "reason": "Sharp object within a child's reach.",
      "recommendation": "Store the knife in a secure location."
    }
  ]
}
```

---

## Running Locally

### Option 1 – Python

Clone the repository.

```bash
git clone <repository-url>
cd room-safety
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

Windows

```bash
.venv\Scripts\activate
```

macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Start the API.

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

### Option 2 – Docker

Build the Docker image.

```bash
docker build -t room-safety .
```

Run the container.

```bash
docker run -p 8000:8000 room-safety
```

---

## Current Limitations

- Currently optimized for indoor environments.
- Focuses only on safety rules for children aged 2–6.
- Uses predefined safety rules instead of a learned risk model.
- No web interface yet.

---

## Roadmap

- [ ] Build a frontend application
- [ ] Deploy the API to the cloud
- [ ] Add dangerous corner detection
- [ ] Expand the rule set for more household hazards
- [ ] Support additional target groups

---

## Motivation

This project was built to explore the combination of modern computer vision models with a practical rule-based decision system. Instead of only detecting objects, the goal is to provide meaningful safety information that can be consumed through a simple API.

---

## Acknowledgements

This project uses the following open-source models:

- Grounding DINO
- SAM2
- FastAPI
- Hugging Face Transformers
- PyTorch

  <img width="965" height="446" alt="test" src="https://github.com/user-attachments/assets/381bcbb4-009e-4e61-b381-06dd9cfdc1b5" />
<img width="965" height="446" alt="image" src="https://github.com/user-attachments/assets/4de2ab2b-6805-47ca-be5c-d9b4ee2ae3e3" />

