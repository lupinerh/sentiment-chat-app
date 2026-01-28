import logging
from google import genai
from google.genai import types
import os
from typing import Generator, List, Dict, Any, Optional

ChatMessage = Dict[str, Any]

class GeminiChatbot: 

    DEFAULT_SYSTEM_PROMPT_EN = (
        "You are a witty and supportive AI companion. Your primary goal "
        "is to engage in a natural, lighthearted conversation. "
        "You MUST respond in Russian. "
        "Keep your answers short, strictly 1-2 sentences, "
        "and always complete your thought. "
        "The user's message will contain their sentiment "
        "and a numerical score, formatted like: '[Sentiment: LABEL (SCORE)]'. "
        "Acknowledge or comment on this sentiment score in your response "
        "ONLY IF the sentiment is notably strong "
        "(e.g., very positive or very negative, like above +0.6 or below -0.6)"
        "OR if commenting on it feels particularly natural "
        "and relevant to the flow of conversation. "
        "If you do comment, you could say something like "
        "'Вижу, ваше настроение на +0.7! ' "
        "or 'Понимаю, -0.8, это непросто... '. "
        "Do NOT mechanically repeat the score in every single message. "
        "Prioritize a engaging chat."
    )
    
    def __init__(self, 
                 api_key: str, 
                 model_name: str,
                 temperature: float = 0.7,
                 max_output_tokens: int = 150
                ) -> None:

        if not api_key:
            logging.error("API для Gemini не предоставлен")
            self.client = None
            return

        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            logging.error(f"Не удалось создать GeminiChatbot")
            self.client = None
            return

        self.model_name = model_name
        self.temperature=temperature
        self.max_output_tokens=max_output_tokens
        self.system_prompt=GeminiChatbot.DEFAULT_SYSTEM_PROMPT_EN


    def _prepare_contents(
            self, 
            chat_history: List[ChatMessage],
            current_user_message_with_sentiment: str
        ) -> List[types.Content]:

        contents = []
        
        for entry in chat_history:
            role = "user" if entry["role"] == "user" else "model"
            contents.append(types.Content(parts=[types.Part(text=entry["content"])], role=role))
        
        contents.append(types.Content(parts=[types.Part(text=current_user_message_with_sentiment)], role="user"))
        
        return contents
    
    @property
    def is_ready(self) -> bool:
        return self.client is not None

    def generate_response(self, 
                          user_message: str, 
                          user_sentiment_label: str, 
                          user_sentiment_score: float, 
                          chat_history: List[ChatMessage]
                          ) -> Generator[str, None, None]:
        
        if not self.client:
            yield f" [Ошибка: Модель Gemini не инициализирована.] "
            return

        current_message_text = \
            f"[Sentiment: {user_sentiment_label} ({user_sentiment_score:.2f})]{user_message}"
        
        # Default settings (Gemini)
        config_system_instruction = self.system_prompt
        final_user_message = current_message_text

        # Gemma
        if "gemma" in self.model_name.lower():
            config_system_instruction = None
            final_user_message = \
                f"System Instruction: {self.system_prompt}\n\nUser Message: {current_message_text}"
        
        contents_for_api = self._prepare_contents(
            chat_history, 
            final_user_message
        )
        
        try:

            response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents_for_api,
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    system_instruction=config_system_instruction
                    )
            )

            for chunk in response:
                if chunk.text: 
                    yield chunk.text
                
        except Exception:
            logging.error("Ошибка генерации ответа Gemini")
            yield f" [Произошла ошибка в Gemini. Попробуй еще раз] "
