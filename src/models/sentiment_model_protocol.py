from typing import Protocol


class SentimentModelProtocol(Protocol):
    def predict(self, text: str) -> float:
        ...

    def explain_shap_text(self, text: str) -> list[tuple[str, float]]:
        ...
