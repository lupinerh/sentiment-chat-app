import logging
import re
import json
import pandas as pd
from typing import List

class Preprocessor:
    TEXT_PATTERNS = [
        (r'(?:http|https)?://\S+\b|www\.\S+', ' '), # URL

        (r'@\w+', ' '), # username
        (r'\S*@\S*\s?', ' '), # email
        (r'\brt\b', ' '), # retweets
        (r'#(\w+)', r'\1 '),  # hastags - only #
        (r'\d+', ' '), # numbers
        
        # smiles and emoji
        (r"\(\(+", " "), # (((
        (r"\)+", " "), # )))
        
        (r'([>:;=-]+[\(cсCС/\\[L{<@]+)', ' '),  # :(, >:(
        (r'([>:;=-]+[\)dD*pPрРbB]+)', ' '),    # :), :-D
        (r'\b(99+|0_0|o_o|о_о)\b', ' '), # text bad smiles
        (r'\b(:d+|xd+|:dd|dd)\b', ' '), # text good smiles
        (r'\b[ах]{2,}[ах]*\b', ' <LAUGH> '), 
        (r'[\U0001F600-\U0001F64F]', ' '), # emoji
        
        (r'[\r\n\t]', ' '), # newline/tabs
        (r'-\s+', ' '),
        (r'[^\w\s<>]', ' '),  # delete punct exept < >
        (r'(\w)\1{2,}', r'\1\1 '),  # duplicates
    ]
    
    def __init__(self, stopwords_path: str) -> None: 
        self.stopwords_path = stopwords_path

        self.text_patterns = None
        self.stopwords = None
        
        self._load_resources()

    def _load_resources(self) -> None:
        # patterns
        self.text_patterns = Preprocessor.TEXT_PATTERNS
        # stopwords
        with open(self.stopwords_path, 'r', encoding='utf-8') as f:
            self.stopwords = set(json.load(f)) 
        logging.info(f"Стоп-слова успешно загружены")

    def preprocess(self, text: str) -> str:
        prepr_text = text.lower().replace('ё', 'е')
        
        for pattern, replacement in self.text_patterns:
            prepr_text = re.sub(pattern, replacement, prepr_text)
    
        prepr_text = re.sub(r'\s+', ' ', prepr_text).strip()

        if not prepr_text:
            return ""
        
        prepr_tokens = prepr_text.split()
        prepr_tokens = [token for token in prepr_tokens 
                        if len(token) > 1 and token not in self.stopwords]

        prepr_text = " ".join(prepr_tokens) 
        
        return prepr_text
        
    def preprocess_batch(self, texts: List[str]) -> List[str]:

        text_series = pd.Series(texts)
        
        prepr_texts = text_series.str.lower().str.replace('ё', 'е', 
                                                          regex=False)
        
        for pattern, replacement in self.text_patterns:
            prepr_texts = prepr_texts.str.replace(pattern, 
                                                  replacement, 
                                                  regex=True)
        
        prepr_texts = prepr_texts.str.replace(r'\s+', ' ', 
                                              regex=True).str.strip()
        
        prepr_tokens = prepr_texts.str.split()
    
        def process_tokens(tokens):
            prepr_tokens = [token for token in tokens 
                            if len(token) > 1 and token not in self.stopwords]
            return ' '.join(prepr_tokens)
            
        prepr_texts = prepr_tokens.apply(process_tokens)
        
        return list(prepr_texts.values)
