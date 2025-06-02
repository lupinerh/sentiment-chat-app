import logging
import streamlit as st
import time
from src.models.logreg_classifier import LogRegClassifier 
from src.sentiment_analysis import (analyze_text_sentiment, 
                                    get_sentiment_parameters)
from src.app_state import add_message_to_chat_history
from src.config_and_settings import SessionKeys

def process_user_send_action(user_draft_text: str, 
                             sentiment_model: LogRegClassifier) -> None:
    """Processes the user's action of sending a message."""
    if not user_draft_text.strip():
        st.toast("Пожалуйста, напишите что-нибудь перед отправкой", icon="✍️")
        return

    final_score = analyze_text_sentiment(user_draft_text, sentiment_model)
    final_label, _ = get_sentiment_parameters(final_score)
    shap_values = sentiment_model.explain_shap_text(user_draft_text)

    add_message_to_chat_history("user", user_draft_text, final_score, 
                                final_label, shap_values)

    st.session_state[SessionKeys.USER_DATA_FOR_BOT] = {
        "content": user_draft_text,
        "label": final_label,
        "score": final_score
    }
    
    st.session_state[SessionKeys.BOT_IS_TYPING] = True
    st.session_state[SessionKeys.USER_DRAFT_INPUT] = ""

    st.rerun()


def handle_bot_response_generation(sentiment_model: LogRegClassifier, 
                                   llm_chatbot) -> None:
    """
    Generates the bot's response, displays it streamingly,
    and adds it to history.
    """
    user_data_for_bot = st.session_state[SessionKeys.USER_DATA_FOR_BOT]

    full_bot_response_text = ""
 
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Бот печатает..."):
            try:
                chat_history_for_llm = (
                    st.session_state[SessionKeys.CHAT_HISTORY][:-1]
                )

                response_generator = llm_chatbot.generate_response(
                    user_message=user_data_for_bot["content"],
                    user_sentiment_label=user_data_for_bot["label"],
                    user_sentiment_score=user_data_for_bot["score"],
                    chat_history=chat_history_for_llm
                )

                for chunk in response_generator:
                    full_bot_response_text += chunk
                    message_placeholder.markdown(full_bot_response_text + "▌")
                    time.sleep(0.01) # for typing effect
            except Exception as e:
                logging.error(f'Ошибка генерации ответа')
                full_bot_response_text = ("Извините, у меня возникла ошибка "
                                          "при генерации ответа.")
            finally:
                message_placeholder.markdown(full_bot_response_text)

        bot_score = analyze_text_sentiment(full_bot_response_text, 
                                           sentiment_model)
        bot_label, _ = get_sentiment_parameters(bot_score)
        bot_shap_values = sentiment_model.explain_shap_text(
            full_bot_response_text
        )

        add_message_to_chat_history(role="assistant",
                                    content=full_bot_response_text,
                                    sentiment_score=bot_score,
                                    sentiment_label=bot_label,
                                    shap_scores=bot_shap_values)
    
