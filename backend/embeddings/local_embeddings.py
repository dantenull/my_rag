from .base import Embeddings
from pathlib import Path
# from transformers import AutoTokenizer, PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Union


class LocalEmbeddings(Embeddings):
    def __init__(self, path: str, **kw) -> None:
        self.embeddings_path = path 
        self.embedding_name = Path(path).name
        self.embedding_type = 'local'
        # self.embeddings: PreTrainedTokenizer = AutoTokenizer.from_pretrained(path)
        self.embeddings: SentenceTransformer = SentenceTransformer(path)
    
    def encode(self, inputs: Union[str, List], **kw) -> List:
        if isinstance(inputs, str):
            return [self.embeddings.encode(inputs, normalize_embeddings=True, **kw)]
        elif isinstance(inputs, list):
            return [self.embeddings.encode(input, normalize_embeddings=True, **kw) for input in inputs]
        return []
