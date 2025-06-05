import logging
import os
import torch
import numpy as np
import shap
from typing import List, Optional, Tuple

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.preprocessors.bert_preprocessor import BertPreprocessor 

class BertClassifier:
    def __init__(self,
                 preprocessor: BertPreprocessor,
                 model_dir: str,
                 class_names: Optional[List[str]] = None,
                 device: Optional[str] = None
                ) -> None:
        
        self.preprocessor = preprocessor

        self.model_dir = model_dir
        self.class_names = class_names or ['negative', 'positive']

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cpu')
        
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None

        self._load_model_and_tokenizer()


    def _load_model_and_tokenizer(self) -> None:
        if not os.path.isdir(self.model_dir):
            logging.error(f"Директория модели не найдена: {self.model_dir}")
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self.model = \
            AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"""Модель и токенизатор успешно загружены 
                         из: {self.model_dir}""")
        except Exception as e:
            logging.error(f"""Ошибка при загрузке модели/токенизатора 
                          из '{self.model_dir}'""")
            self.model = None
            self.tokenizer = None

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _predict_proba_batch(self, texts: List[str]) -> np.ndarray:

        inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        input_ids_batch = inputs['input_ids'].to(self.device)
        attention_mask_batch = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids_batch, 
                                 token_type_ids=None, 
                                 attention_mask=attention_mask_batch)
        
        probabilities = torch.softmax(outputs.logits, dim=1)
        return probabilities.cpu().numpy()

    def predict(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        
        # preproccess
        prepr_text = self.preprocessor.preprocess(text)
        if not prepr_text or not prepr_text.strip():
            return 0.0

        proba = self._predict_proba_batch([text])[0]
        pos_proba = proba[1]
        
        # [0, 1] -> [-1, 1]
        pos_proba = 2 * pos_proba - 1
        
        return pos_proba

    def _predict_function_shap(self, texts: List[str]) -> np.ndarray:
        prepr_texts = self.preprocessor.preprocess_batch(texts)
        proba = self._predict_proba_batch(prepr_texts)
        return proba
    
    def explain_shap_text(self, text: str) -> List[Tuple[str, float]]:
        if len(text.split()) < 2:
            logging.info('Explain невозможен: нужно больше слов')
            return []

        mask_token = self.tokenizer.mask_token
        shap_masker = shap.maskers.Text(self.tokenizer, mask_token=mask_token)
        
        explainer = shap.Explainer(self._predict_function_shap, 
                                   shap_masker, 
                                   output_names=self.class_names)
        
        shap_values = explainer([text])

        tokens = shap_values.data[0]

        class_explain = 1
        shap_values_class = shap_values.values[0, :, class_explain]

        shap_scores = list(zip(tokens, shap_values_class))

        return shap_scores
