import logging
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from src.models.sentiment_model_protocol import SentimentModelProtocol
from src.sentiment_analysis import (
    analyze_text_sentiment,
    get_sentiment_parameters,
    display_shap_annotated_text
)
from src.config_and_settings import (
    SessionKeys, WELCOME_TITLE, WELCOME_SUBHEADER, 
    WELCOME_EXAMPLES_HEADER, WELCOME_EXAMPLES,
    SELECTED_BUTTON_CSS, UNSELECTED_BUTTON_CSS,
    MODEL_ID_LOGREG, MODEL_ID_BERT
)

def configure_page() -> None:
    """Configures the Streamlit page settings."""
    st.set_page_config(
        page_title="Чат с Анализом Настроения",
        layout="wide",
        page_icon="💬"
    )

def display_sentiment_model_selector():

    if "ui_selected" not in st.session_state:
        st.session_state.ui_selected = MODEL_ID_LOGREG

    logreg_css = (SELECTED_BUTTON_CSS 
                  if st.session_state.ui_selected == MODEL_ID_LOGREG 
                  else UNSELECTED_BUTTON_CSS)
    bert_css = (SELECTED_BUTTON_CSS 
                if st.session_state.ui_selected == MODEL_ID_BERT 
                else UNSELECTED_BUTTON_CSS)

    st.caption("Анализатор настроения:")
    col1, col2 = st.columns(2)

    with col1:
        with stylable_container(key=f"{MODEL_ID_LOGREG}_container", 
                                css_styles=logreg_css):
            if st.button("⚡️ **ML:** Быстрее", 
                         use_container_width=True, 
                         key=f"{MODEL_ID_LOGREG}_btn"):
                if st.session_state.ui_selected != MODEL_ID_LOGREG:
                    st.session_state[SessionKeys.SELECTED_SENTIMENT_MODEL] = \
                            st.session_state[SessionKeys.SENTIMENT_MODELS_DICT][MODEL_ID_LOGREG]
                    st.session_state.ui_selected = MODEL_ID_LOGREG
                    logging.info(f"Sentiment-модель изменена LOGREG")
                    st.rerun()

    with col2:
        with stylable_container(key=f"{MODEL_ID_BERT}_container", 
                                css_styles=bert_css):
            if st.button("📊 **Bert:** Точнее", 
                         use_container_width=True, 
                         key=f"{MODEL_ID_BERT}_btn"):
                if st.session_state.ui_selected != MODEL_ID_BERT:
                    st.session_state[SessionKeys.SELECTED_SENTIMENT_MODEL] = \
                            st.session_state[SessionKeys.SENTIMENT_MODELS_DICT][MODEL_ID_BERT]
                    st.session_state.ui_selected = MODEL_ID_BERT
                    logging.info(f"Sentiment-модель изменена BERT")
                st.rerun()


def display_compare_toggle() -> bool:
    st.caption("Выбранная модель влияет на чат.")
    return st.toggle("Сравнить модели",
                     value=st.session_state.get(SessionKeys.COMPARE_MODE, False),
                     key=SessionKeys.COMPARE_MODE)


def display_compact_compare_card(logreg_score: float,
                                 bert_score: float) -> None:
    container_css = (
        "{background-color: #f7f7f7; padding: 10px 12px; "
        "border-radius: 10px; margin: 8px 0; "
        "border: 1px solid rgba(0,0,0,0.06);}"
    )

    def render_row(model_label: str,
                   score: float,
                   slider_key: str) -> None:
        emoji_label, color = get_sentiment_parameters(score)
        col_label, col_score, col_slider = st.columns([2, 2, 4])

        with col_label:
            with stylable_container(
                css_styles=(
                    f"{{background-color: {color}; padding: 4px 8px; "
                    f"border-radius: 999px; display: inline-block;}}"
                ),
                key=f"{slider_key}_badge"
            ):
                st.markdown(f"**{model_label}**")

        with col_score:
            st.markdown(f"{emoji_label}  **{score:+.2f}**")

        with col_slider:
            st.slider("Level", -1.0, 1.0, float(score), 0.01,
                      disabled=True, label_visibility="collapsed",
                      key=slider_key)

    with stylable_container(css_styles=container_css,
                            key="compare_compact_container"):
        render_row("⚡️ ML", logreg_score, "compare_slider_ml")
        render_row("📊 BERT", bert_score, "compare_slider_bert")


def _display_word_explanations(model_label: str,
                               text: str,
                               sentiment_model: SentimentModelProtocol) -> None:
    st.markdown(f"**{model_label}: вклад слов**")

    if len(text.split()) < 2:
        st.caption("Введите минимум 2 слова, чтобы увидеть вклад слов.")
        return

    with st.spinner("Анализ важности слов..."):
        shap_scores = sentiment_model.explain_shap_text(text)

    if shap_scores:
        display_shap_annotated_text(shap_scores)
    else:
        st.caption("Не удалось вычислить вклад слов для этого текста.")


def display_compare_input_sentiment_analysis(text: str,
                                             model_logreg: SentimentModelProtocol,
                                             model_bert: SentimentModelProtocol
                                             ) -> None:
    st.caption("Сравнение моделей по одному тексту")
    st.caption("Сравнение может быть медленнее на больших текстах.")

    logreg_score = analyze_text_sentiment(text, model_logreg)
    bert_score = analyze_text_sentiment(text, model_bert)

    score_diff = bert_score - logreg_score
    if abs(score_diff) < 0.01:
        st.markdown("Модели дают почти одинаковую оценку.")
    elif score_diff > 0:
        st.markdown(f"**BERT** более позитивен на **{score_diff:+.2f}**")
    else:
        st.markdown(f"**ML** более позитивен на **{abs(score_diff):.2f}**")

    display_compact_compare_card(logreg_score, bert_score)

    _display_word_explanations("⚡️ ML", text, model_logreg)
    _display_word_explanations("📊 BERT", text, model_bert)


def display_current_input_sentiment_analysis(score: float,
                                             text: str,
                                             sentiment_model: SentimentModelProtocol
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
