import streamlit as st
from streamlit_extras.annotated_text import annotated_text
from src.models.logreg_classifier import LogRegClassifier
from src.config_and_settings import SENTIMENT_THRESHOLD

def analyze_text_sentiment(text: str, 
                           sentiment_model: LogRegClassifier) -> float:
    """Analyzes text sentiment and returns a score."""
    if not text.strip():
        return 0.0
    return sentiment_model.predict(text)


def get_sentiment_parameters(score: float, 
                             threshold: float = SENTIMENT_THRESHOLD
                             ) -> tuple[str, str]:
    """Returns a label and color for the sentiment score."""
    if score > 0 + 3 * threshold:
        return "😊 Позитивнее", "#bbffbb"
    elif score > 0 + threshold:
        return "🙂 Позитивная", "#ccffcc"
    elif score < 0 - 3 * threshold:
        return "😞 Негативнее", "#ffbbbb"
    elif score < 0 - threshold:
        return "🙁 Негативная", "#ffcccc"
    else:
        return "😐 Нейтральная", "#f0f0f0"


def format_shap_annotation(token: str, 
                           score: float, 
                           max_abs_score: float) -> tuple:
    """Formats a token for annotated_text based on its SHAP value."""
    threshold_color = 0.01
    label_tooltip = f"{score:+.2f}"

    intensity = (
        min(abs(score) / max_abs_score, 1.0) if max_abs_score != 0 else 0.0
    )
    min_alpha, max_alpha_range = 0.2, 0.8
    alpha = min_alpha + intensity * max_alpha_range

    if score > threshold_color:
        background_color = f"rgba(144, 238, 144, {alpha})" # green
    elif score < -threshold_color:
        background_color = f"rgba(255, 182, 193, {alpha})" # red
    else:
        return (token, "", "rgba(240, 240, 240, 0.3)") # grey
    return (token, label_tooltip, background_color)


def display_shap_annotated_text(shap_scores: list | None):
    """Displays annotated text based on SHAP values."""
    if shap_scores:
        abs_scores = [abs(s) for _, s in shap_scores]
        max_abs_score = max(abs_scores) if abs_scores else 0
        display_parts = []
        for token, score_val in shap_scores:
            formatted_annotation = format_shap_annotation(token,
                                                            score_val,
                                                            max_abs_score)
            display_parts.append(formatted_annotation)
        if display_parts:
            annotated_text(*display_parts)
