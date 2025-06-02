import logging

import streamlit as st
from streamlit_extras.st_keyup import st_keyup

from src.config_and_settings import (SessionKeys, 
                                     EMPTY_TEXT_PLACEHOLDER, 
                                     load_config)
from src.model_loader import (load_sentiment_model_cached, 
                              load_llm_chatbot_cached)
from src.sentiment_analysis import analyze_text_sentiment
from src.ui_components import (
    configure_page, 
    display_current_input_sentiment_analysis,
    display_welcome_message,
    display_chat_history
)
from src.app_state import initialize_session_state
from src.chat_logic import (process_user_send_action, 
                            handle_bot_response_generation)

from src.models.logreg_classifier import LogRegClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# -- UI Column Rendering --
def render_left_column(sentiment_model: LogRegClassifier) -> None:
    """Renders the left UI column (user input, sentiment analysis)."""
    st.caption(
        "Введите текст – тональность обновляется сразу.  \n"
        "Больше одного слова – обновляется анализ и по словам."
    )

    bot_is_typing_now = st.session_state.get(SessionKeys.BOT_IS_TYPING, False)
    current_draft_from_state = \
        st.session_state.get(SessionKeys.USER_DRAFT_INPUT, "")

    text_for_analysis = st.session_state.get(SessionKeys.USER_DRAFT_INPUT, "")
    draft_sentiment_score = analyze_text_sentiment(text_for_analysis, 
                                                   sentiment_model)

    display_current_input_sentiment_analysis(draft_sentiment_score, 
                                             text_for_analysis, 
                                             sentiment_model)

    user_input_from_keyup = st_keyup(
        label="",
        placeholder=EMPTY_TEXT_PLACEHOLDER,
        value=current_draft_from_state,
        debounce=300,
        key="user_input_keyup_chat_left_col",
        disabled=bot_is_typing_now,
        )

    if not bot_is_typing_now:
        st.session_state[SessionKeys.USER_DRAFT_INPUT] = user_input_from_keyup

    if st.button("✉️ Отправить в чат",
                 key="send_to_chat_button_left_col",
                 use_container_width=True,
                 disabled=bot_is_typing_now or not text_for_analysis.strip()):
        process_user_send_action(
            st.session_state.get(SessionKeys.USER_DRAFT_INPUT, ""),
            sentiment_model
        )


def render_right_column(sentiment_model: LogRegClassifier, 
                        llm_chatbot) -> None:
    """Renders the right UI column (chat history, bot response)."""
    st.caption("Модель учитывает тональность вашего сообщения при ответе.")

    chat_display_area = st.container(height=500, border=False)
    with chat_display_area:
        display_welcome_message()
        display_chat_history(st.session_state.get(SessionKeys.CHAT_HISTORY, 
                                                  []))

        user_has_sent_message = \
            st.session_state.get(SessionKeys.USER_DATA_FOR_BOT) is not None
        bot_should_be_typing = \
            st.session_state.get(SessionKeys.BOT_IS_TYPING, False)

        if user_has_sent_message and bot_should_be_typing:
            handle_bot_response_generation(sentiment_model, llm_chatbot)

            st.session_state[SessionKeys.USER_DATA_FOR_BOT] = None
            st.session_state[SessionKeys.BOT_IS_TYPING] = False
            st.rerun()


# -- Main Application Function --
def main_app() -> None:
    """Main application function: initialization and execution."""
    configure_page()
    initialize_session_state()

    app_config = load_config()
    if not app_config:
        st.error("Ошибка загрузки данных")
        return

    if not st.session_state.get(SessionKeys.MODELS_LOADED, False):
        st.session_state[SessionKeys.SENTIMENT_MODEL] = \
            load_sentiment_model_cached(app_config)
        st.session_state[SessionKeys.LLM_CHATBOT] = \
            load_llm_chatbot_cached(app_config)

        if (st.session_state[SessionKeys.SENTIMENT_MODEL] 
            and st.session_state[SessionKeys.LLM_CHATBOT]):
            st.session_state[SessionKeys.MODELS_LOADED] = True
            st.rerun()
        elif not st.session_state[SessionKeys.SENTIMENT_MODEL]:
            err = "Sentiment Model не загружена, приложение остановлено"
            st.error(err)
            logging.error(err)
            st.stop()
        elif not not st.session_state[SessionKeys.MODELS_LOADED]:
            st.error("LMM model не загружена, приложение остановлено")
            logging.error("LMM model не загружена, приложение остановлено")
            st.stop()

    sentiment_model = st.session_state.get(SessionKeys.SENTIMENT_MODEL)
    llm_chatbot = st.session_state.get(SessionKeys.LLM_CHATBOT)

    left_column, right_column = st.columns([2, 3])

    with left_column:
        render_left_column(sentiment_model)

    with right_column:
        render_right_column(sentiment_model, llm_chatbot)

if __name__ == "__main__":
    main_app()
