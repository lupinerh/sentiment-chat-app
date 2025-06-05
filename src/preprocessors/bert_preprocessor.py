import re
import pandas as pd
from typing import List

class BertPreprocessor:
    TEXT_PATTERNS = [
        (r'(?:http|https)?://\S+\b|www\.\S+', ' '), # URL

        (r'@\w+', ' '), # username
        (r'\S*@\S*\s?', ' '), # email
        (r'(?i)\brt\b', ' '), # retweets
        
        # smiles and emoji
        (r"\(\(+", " "), # (((
        (r"\)+", " "), # )))
        
        (r"(?i)([>:;=\-]+[\(c/\\[l{<@]+)", " "),  # :(. >:(
        (r"(?i)([>:;=\-]+[\)d*p]+)",       " "),  # :), :-D
        
        (r"(?i)\b(99+|0_0|o_o|о_о)\b",     " "),  # text bad smiles
        (r"(?i)\b(:d+|xd+|:dd|dd)\b",       " "),  # text good smiles
        
        (r'[\U0001F600-\U0001F64F]', ' '), # emoji
        
    ]
    
    def __init__(self) -> None: 
        self.text_patterns = None
        self.text_patterns = BertPreprocessor.TEXT_PATTERNS

    def preprocess(self, text: str) -> str:        
        for pattern, replacement in self.text_patterns:
            prepr_text = re.sub(pattern, replacement, text)
        return prepr_text
        
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        text_series = pd.Series(texts)     
        for pattern, replacement in self.text_patterns:
            prepr_texts = text_series.str.replace(pattern, 
                                                  replacement, 
                                                  regex=True)        
        return list(prepr_texts.values)
