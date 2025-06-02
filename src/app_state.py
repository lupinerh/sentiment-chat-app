import streamlit as st
from src.config_and_settings import SessionKeys

def initialize_session_state() -> None:
    """Initializes the session state with default values."""
    default_values = {
        SessionKeys.CHAT_HISTORY: [],
        SessionKeys.USER_DRAFT_INPUT: "",
        SessionKeys.USER_DATA_FOR_BOT: None,
        SessionKeys.MODELS_LOADED: False,
        SessionKeys.SENTIMENT_MODEL: None,
        SessionKeys.LLM_CHATBOT: None,
        SessionKeys.BOT_IS_TYPING: False,
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_message_to_chat_history(role: str,
                                content: str,
                                sentiment_score: float,
                                sentiment_label: str,
                                shap_scores: list | None) -> None:
    """Adds a message to the chat history in the session state."""
    st.session_state[SessionKeys.CHAT_HISTORY].append({
        "role": role,
        "content": content,
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "shap_scores": shap_scores
    })
