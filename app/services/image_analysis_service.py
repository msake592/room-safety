from app.models.grounding_dino import GroundingDinoDetector
from app.models.sam2 import Sam2Segmenter
from app.risk_engine.engine import RiskEngine


class ImageAnalysisService:
    def __init__(self):
        self.detector = GroundingDinoDetector()
        self.segmenter = Sam2Segmenter()
        self.risk_engine = RiskEngine()

    def analyze(self, image, labels: list[str]) -> dict:
        detections = self.detector.detect(image, labels)

        boxes = [
            detection["box"]
            for detection in detections
        ]

        masks = self.segmenter.segment(image, boxes)
        risks = self.risk_engine.evaluate(detections)

        return {
            "detections": detections,
            "masks": masks,
            "risks": risks,
        }