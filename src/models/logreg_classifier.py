import logging
import os
import shap
import joblib
import numpy as np
from typing import List, Optional, Tuple

from ..preprocessor import Preprocessor


class LogRegClassifier:
    
    def __init__(self, 
                 preprocessor: Preprocessor, 
                 model_path: str, 
                 class_names: Optional[List[str]] = None) -> None:
        self.preprocessor = preprocessor
        self.model_path = model_path
        self.class_names = class_names or ['negative', 'positive']
        
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logging.info('Sentiment Model Не загружена')
            return None
        else:
            model = joblib.load(self.model_path)
            logging.info('Sentiment Model загружена')
            return model
        
    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, text: str) -> float:
        
        if not text.strip():
            return 0.0
            
        prepr_text = self.preprocessor.preprocess(text)
        
        if not prepr_text.strip():
            return 0.0
            
        proba = self.model.predict_proba([prepr_text])[0]
        pos_proba = proba[1]

        # [0, 1] -> [-1, 1]
        pos_proba = 2 * pos_proba - 1
        
        return pos_proba

    def _predict_function_shap(self, texts: List[str]) -> np.ndarray:
        prepr_texts = self.preprocessor.preprocess_batch(texts)
        proba = self.model.predict_proba(prepr_texts)
        return proba

    def explain_shap_text(self, text: str) -> List[Tuple[str, float]]:
        if len(text.split()) < 2:
            logging.info('Explain невозможен: нужно больше слов')
            return []
        
        masker = shap.maskers.Text(tokenizer=r"\W+")
        
        explainer = shap.Explainer(
            model=self._predict_function_shap,
            masker=masker,
            output_names=self.class_names
        )
        
        shap_values = explainer([text])

        tokens = shap_values.data[0]

        class_explain = 1
        shap_values_class = shap_values.values[0, :, class_explain]

        shap_scores = list(zip(tokens, shap_values_class))

        return shap_scores
