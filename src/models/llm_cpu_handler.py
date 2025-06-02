import logging
import os
import time
from llama_cpp import Llama
import streamlit as st
from huggingface_hub import hf_hub_download
from typing import Generator, List, Dict, Any, Optional

ChatMessage = Dict[str, Any]

class LLMCPUChatbot:

    DEFAULT_SYSTEM_PROMPT_EN = (
        "You are friendly and witty AI companion. "
        "Your primary goal is to engage in lighthearted "
        "and supportive conversation."
        "Always respond in Russian. "
        "Keep your answers short, strictly 1-2 sentences"
        "And always complete your thought. "
        "Always consider the user's message sentiment and its numerical score,"
        "which is provided in the message "
        "Briefly acknowledge this sentiment in your response. "
    )

    def __init__(self, 
                 model_local_dir: str, 
                 model_url_repo: str,
                 model_filename: str) -> None:

        self.model_local_dir = model_local_dir
        self.model_url_repo = model_url_repo
        self.model_filename = model_filename
        self.model_local_path = os.path.join(self.model_local_dir, 
                                             self.model_filename)
        self.max_tokens = 120
        self.temperature = 0.7
        self.top_p = 0.9
        self.stop_seq = ["<|end|>", 
                         "\n<|user|>", 
                         "\n<|system|>",
                         "<|endoftext|>",
                         "\n"]
        self.system_prompt = LLMCPUChatbot.DEFAULT_SYSTEM_PROMPT_EN

        self.model = self._load_model()
    
    def _download_model(self):
        hf_hub_download(repo_id=self.model_url_repo,
                        filename=self.model_filename,
                        local_dir=self.model_local_dir,
                        local_dir_use_symlinks=False)
    
    def _load_model(self) -> Optional[Llama]:

        if not os.path.exists(self.model_local_path):
            if self.model_url_repo and self.model_filename:
                st.info(f"Модель '{self.model_local_path}' не найдена. "
                        f"Загрузка с Hugging Face Hub "
                        f"(репозиторий: {self.model_url_repo})")
                logging.info(f"Модель не найдена: {self.model_local_path}. "
                            f" Загрузка с Hugging Face Hub..")
                try:
                    self._download_model()
                    st.success(f"Модель c URL загружена в "
                               f"'{self.model_local_path}'.")
                except Exception as e:
                    logging.error(f"Загрузка модели не удалась")
                    return None
            else:
                error_msg = (f"Модель не найдена "
                             f"и URL не предоставлен.")
                st.error(error_msg)
                logging.error(error_msg)
                return None
        else:
            logging.info(f"Модель найдена: {self.model_local_path}.")

        start_time = time.time()
        try:
            model = Llama(
                model_path=self.model_local_path,
                n_ctx=4096,
                n_gpu_layers=0, # cpu
                n_threads=max(os.cpu_count() // 2, 1),
                n_batch=128,
                verbose=False
            )
            end_time = time.time()
            logging.info(
                f"LLM загружена за {end_time - start_time:.2f} секунд."
                )
            return model
        except Exception as e:
            error_message = f"Ошибка при загрузке локальной LLM"
            st.error(error_message)
            logging.error(error_message)
            return None
        
    def _format_prompt(self, 
                       user_message: str, 
                       user_sentiment_label: str, 
                       user_sentiment_score: float, 
                       chat_history: List[ChatMessage], 
                       system_prompt: str) -> str:
        
        prompt = f"<|system|>\n{system_prompt}<|end|>\n"
        for entry in chat_history:
            role = entry["role"]
            content = entry["content"]
            sentiment_info_str = ""
            if role == "user" and "sentiment_label" in entry:
                s_label = entry.get("sentiment_label")
                s_score = entry.get("sentiment_score")
                sentiment_info_str = f"[Sentiment: {s_label} ({s_score:.2f})] "
            prompt += f"<|{role}|>\n{sentiment_info_str}{content}<|end|>\n"
        
        current_user_sentiment_info_str = ""
        current_user_sentiment_info_str = f"""
            [Sentiment: {user_sentiment_label} ({user_sentiment_score:.2f})] 
            """
            
        prompt += f"""<|user|>\n{current_user_sentiment_info_str}
                      {user_message}<|end|>\n<|assistant|>\n"""
        return prompt
    
    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def generate_response(self, 
                          user_message: str, 
                          user_sentiment_label: str, 
                          user_sentiment_score: float, 
                          chat_history: List[ChatMessage]
                          ) -> Generator[str, None, None]:
        
        if self.model is None:
            st.warning("LLM не загружена, ответ не может быть сгенерирован.")
            yield "Извините, возникли технические сложности. Попробуйте позже."
            return

        formatted_prompt = self._format_prompt(user_message, 
                                               user_sentiment_label, 
                                               user_sentiment_score, 
                                               chat_history, 
                                               self.system_prompt)
        
        try:
            output_stream = self.model(
                formatted_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                stop=self.stop_seq,
                stream=True,
                echo=False
            )
            for chunk in output_stream:
                token_text = chunk["choices"][0]["text"]
                yield token_text

        except Exception as e:
            st.error(f"Ошибка при генерации ответа LLM")
            logging.error(f"Ошибка при генерации ответа LLM")
            return "Произошла ошибка при обработке запроса LLM."
