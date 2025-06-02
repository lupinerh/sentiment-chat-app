import logging
import os
import yaml
import streamlit as st

from streamlit.errors import StreamlitSecretNotFoundError
from dotenv import load_dotenv

CONFIG_PATH = 'config.yaml'
SENTIMENT_THRESHOLD = 0.2
EMPTY_TEXT_PLACEHOLDER = "Начните вводить текст здесь..."

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
    SENTIMENT_MODEL = "sentiment_model"
    LLM_CHATBOT = "llm_chatbot"
    BOT_IS_TYPING = "bot_is_typing"

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

@st.cache_data
def load_config(config_path: str = CONFIG_PATH) -> dict:
    """Загружает конфигурационный файл YAML."""
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
