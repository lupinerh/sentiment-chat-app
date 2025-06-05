import logging
import os
import yaml
import streamlit as st

from streamlit.errors import StreamlitSecretNotFoundError
from dotenv import load_dotenv

def get_gemini_key():
    api_key = None
    try:
        if "gemini" in st.secrets and "api_key" in st.secrets['gemini']:
            api_key = st.secrets['gemini']['api_key']
            if api_key:
                return api_key
    except StreamlitSecretNotFoundError:
        pass

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key
    
    logging.error("Проблемы с доступом Gemini, приложение не может работать")
    st.error("Проблемы с доступом Gemini, приложение не может работать")
    st.stop()

GEMINI_API_KEY = get_gemini_key()

class SessionKeys:
    CHAT_HISTORY = "chat_history"
    USER_DRAFT_INPUT = "user_draft_input_text"
    USER_DATA_FOR_BOT = "user_data_for_bot"
    MODELS_LOADED = "models_loaded"
    SELECTED_SENTIMENT_MODEL = "selected_sentiment_model"
    SENTIMENT_MODELS_DICT = "sentiment_models_dict"
    LLM_CHATBOT = "llm_chatbot"
    BOT_IS_TYPING = "bot_is_typing"

MODEL_ID_LOGREG = "logreg"
MODEL_ID_BERT = "bert"

CONFIG_PATH = 'config.yaml'
SENTIMENT_THRESHOLD = 0.2
EMPTY_TEXT_PLACEHOLDER = "Начните вводить текст здесь..."

WELCOME_TITLE = "## 💬 Чат с Анализом Настроения"
WELCOME_SUBHEADER = (
    "Я готов к общению и умею понимать эмоции в ваших сообщениях.  \n"
    "Просто введите фразу в поле слева, и вы увидите "
    "анализ тональности в реальном времени.  \n"
    "Когда будете готовы, нажмите '✉️ Отправить в чат', "
    "чтобы начать наш диалог!  \n"
)
WELCOME_EXAMPLES_HEADER = "💡 **Примеры фраз для начала диалога:**"
WELCOME_EXAMPLES = [
    "Привет! Как дела?",
    "У меня такой хороший день сегодня, как твой?",
    "Расскажи что-нибудь о себе."
]

# css style for sentiment model selector
SELECTED_BUTTON_CSS = """
    button {
        border-radius: 12px; padding: 10px 15px !important; font-size: 16px !important;
        font-weight: bold !important; border: 2px solid transparent !important;
        background-color: #D6EAF8 !important; color: black !important; border-color: #AED6F1 !important;
        box-shadow: 0 2px 5px rgba(214, 234, 248, 0.3) !important;
    }
    """
UNSELECTED_BUTTON_CSS = """
    button {
        border-radius: 12px; padding: 10px 15px !important; font-size: 16px !important;
        font-weight: bold !important; border: 2px solid transparent !important;
        background-color: #F0F0F0 !important; color: #A0A0A0 !important; border-color: #E0E0E0 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    }
    """

@st.cache_data
def load_config(config_path: str = CONFIG_PATH) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.error(f"Конфигурационный файл не найден: {config_path}")
        return {}
    except Exception as e:
        st.error(f"Ошибка при загрузке конфигурационного файла: {e}")
        return {}
