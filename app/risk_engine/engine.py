import json
from pathlib import Path


class RiskEngine:
    def __init__(self, rules_path: str = "config/risk_rules.json"):
        self.rules_path = Path(rules_path)
        self.rules = self._load_rules()

    def _load_rules(self) -> dict:
        if not self.rules_path.exists():
            raise FileNotFoundError(
                f"Risk kuralları bulunamadı: {self.rules_path}"
            )

        with self.rules_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def evaluate(self, detections: list[dict]) -> list[dict]:
        results = []

        for detection in detections:
            label = detection["label"].lower().strip()
            rule = self.rules.get(label)

            if rule is None:
                continue

            results.append(
                {
                    **detection,
                    "risk_level": rule["risk_level"],
                    "reason": rule["reason"],
                    "recommendation": rule["recommendation"],
                    "target_group": rule["target_group"],
                }
            )

        return results