import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from src.models.logreg_classifier import LogRegClassifier 
from src.sentiment_analysis import (get_sentiment_parameters, 
                                    display_shap_annotated_text)
from src.config_and_settings import (
    SessionKeys, WELCOME_TITLE, WELCOME_SUBHEADER, 
    WELCOME_EXAMPLES_HEADER, WELCOME_EXAMPLES
)

def configure_page() -> None:
    """Configures the Streamlit page settings."""
    st.set_page_config(
        page_title="Чат с Анализом Настроения",
        layout="wide",
        page_icon="💬"
    )


def display_current_input_sentiment_analysis(score: float, 
                                             text: str, 
                                             sentiment_model: LogRegClassifier
                                             ) -> None:
    """Displays the sentiment analysis of the current input text."""
    emoji_label, color = get_sentiment_parameters(score)
    container_css = (
        f"{{background-color: {color}; padding: 16px; "
        f"border-radius: 8px; margin: 10px 0; "
        f"box-shadow: 0 2px 4px rgba(0,0,0,0.1); "
        f"transition: background-color 0.3s ease-in-out;}}"
    )
    with stylable_container(css_styles=container_css,
                            key='main_sentiment_container_left_col'):
        col_metric, col_slider = st.columns([3, 2])
        with col_metric:
            st.metric(label="Тональность",
                      value=emoji_label,
                      delta=f"{score:.2f}")
        with col_slider:
            st.slider("Level", -1.0, 1.0, float(score), 0.01, 
                      disabled=True, label_visibility="collapsed")

    with st.spinner("Анализ важности слов..."):
        shap_scores = sentiment_model.explain_shap_text(text)
        display_shap_annotated_text(shap_scores)


def display_chat_message_content(message_content: str, 
                                 shap_scores: list | None) -> None:
    """Displays the content of a single chat message."""
    if shap_scores:
        display_shap_annotated_text(shap_scores)
    else:
        st.markdown(message_content)


def display_chat_history(chat_history: list) -> None:
    """Displays the entire chat history."""
    for message in chat_history:
        with st.chat_message(message["role"]):
            display_chat_message_content(message["content"], 
                                         message.get("shap_scores"))
            
            sentiment_label = message.get('sentiment_label', "N/A")
            sentiment_score = message.get('sentiment_score', 0.0)
            st.caption(
                f"Тональность: {sentiment_label} ({sentiment_score:.2f})"
            )


def display_welcome_message() -> None:
    """Displays the welcome message if the chat is empty."""
    history_empty = not st.session_state.get(SessionKeys.CHAT_HISTORY, [])
    bot_not_replying_to_first_message = (
        st.session_state.get(SessionKeys.USER_DATA_FOR_BOT) is None
    )

    if history_empty and bot_not_replying_to_first_message:
        st.markdown(WELCOME_TITLE)
        st.markdown(WELCOME_SUBHEADER)
        st.markdown("---")
        st.markdown(WELCOME_EXAMPLES_HEADER)

        for i, phrase in enumerate(WELCOME_EXAMPLES):
            if st.button(phrase, key=f"example_btn_{i}"):
                st.session_state[SessionKeys.USER_DRAFT_INPUT] = phrase
                st.rerun()
        
        st.markdown("---")
