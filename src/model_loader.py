import logging
import streamlit as st
from src.preprocessor import Preprocessor
from src.models.logreg_classifier import LogRegClassifier
from src.models.llm_cpu_handler import LLMCPUChatbot
from src.models.gemini_handler import GeminiChatbot
from src.config_and_settings import GEMINI_API_KEY # Импортируем API ключ

@st.cache_resource(show_spinner="Загрузка модели настроения...")
def load_sentiment_model_cached(app_config: dict) -> LogRegClassifier | None:
    """Caches and loads the sentiment analysis model."""
    try:
        model_path = app_config['paths']['models']['logreg_classifier']
        stopwords_path = app_config['paths']['preprocessor']['stopwords']
        preprocessor = Preprocessor(stopwords_path=stopwords_path)
        model = LogRegClassifier(preprocessor=preprocessor, 
                                 model_path=model_path)
        
        return model if model.is_ready else None
    
    except KeyError as e:
        logging.error(f"Ошибка конфигурации при загрузке модели настроения")
        st.error(f"Ошибка конфигурации при загрузке модели настроения")
        return None
    except Exception as e:
        logging.error(f"Не удалось загрузить модель настроения")
        st.error(f"Не удалось загрузить модель настроения")
        return None


@st.cache_resource(show_spinner="Загрузка ИИ-модели...")
def load_llm_chatbot_cached(app_config: dict
                            ) -> LLMCPUChatbot | GeminiChatbot | None:
    """Caches and loads the LLM chatbot model."""
    try:
        llm_provider = app_config.get("llm_provider", "llm_cpu")

        if llm_provider == "llm_cpu":
            llm_chatbot_config = \
                app_config['paths']['models']['llm_cpu_chatbot']
            model_local_dir = llm_chatbot_config['model_local_dir']
            model_url_repo = llm_chatbot_config['model_url_repo']
            model_filename = llm_chatbot_config['model_filename']
            model = LLMCPUChatbot(model_local_dir=model_local_dir,
                                  model_url_repo=model_url_repo,
                                  model_filename=model_filename)
            
        elif llm_provider == "gemini_api":
            if not GEMINI_API_KEY:
                st.error("Gemini не найден")
                return None
            gemini_settings = app_config["gemini_api_settings"]
            model_name = gemini_settings["model_name"]
            temperature = gemini_settings["temperature"]
            max_output_tokens = gemini_settings["max_output_tokens"]

            model = GeminiChatbot(
                api_key=GEMINI_API_KEY,
                model_name=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
        else:
            st.error(f"Неизвестный провайдер LLM: {llm_provider}")
            return None
        
        return model if model.is_ready else None
        
    except Exception as e:
        st.error(f"Не удалось загрузить LLM чат-бота")
        return None
