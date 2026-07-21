import json
from pathlib import Path


class RiskEngine:
    def __init__(self, rules_path: str = "config/risk_rules.json"):
        self.rules_path = self._resolve_rules_path(rules_path)
        self.config = self._load_rules()
        self.target_group = self.config.get("target_group", "children_2_6")
        self.rules = self.config.get("rules", [])

    @staticmethod
    def _resolve_rules_path(rules_path: str) -> Path:
        path = Path(rules_path).expanduser()

        if path.is_absolute():
            return path

        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / path

    def _load_rules(self) -> dict:
        if not self.rules_path.exists():
            raise FileNotFoundError(
                f"Risk kuralları bulunamadı: {self.rules_path}"
            )

        with self.rules_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        if not isinstance(config, dict):
            raise ValueError(
                "Risk kuralları JSON kökünde bir nesne olmalıdır."
            )

        rules = config.get("rules")

        if not isinstance(rules, list):
            raise ValueError(
                "Risk kuralları dosyasındaki 'rules' alanı "
                "bir liste olmalıdır."
            )

        return config

    @staticmethod
    def _normalize_label(label: str) -> str:
        return label.lower().strip().rstrip(".")

    def _find_matching_rule(self, detection_label: str) -> dict | None:
        normalized_detection_label = self._normalize_label(
            detection_label
        )

        for rule in self.rules:
            for rule_label in rule.get("labels", []):
                normalized_rule_label = self._normalize_label(
                    rule_label
                )

                if (
                    normalized_detection_label
                    == normalized_rule_label
                    or normalized_rule_label
                    in normalized_detection_label
                ):
                    return rule

        return None

    def evaluate(self, detections: list[dict]) -> list[dict]:
        results = []

        for detection in detections:
            label = detection.get("label")

            if not isinstance(label, str) or not label.strip():
                continue

            rule = self._find_matching_rule(label)

            if rule is None:
                continue

            results.append(
                {
                    **detection,
                    "rule_id": rule.get("id"),
                    "risk_level": rule["risk_level"],
                    "risk_score": rule.get("risk_score"),
                    "reason": rule["reason"],
                    "recommendation": rule["recommendation"],
                    "target_group": self.target_group,
                }
            )

        return results