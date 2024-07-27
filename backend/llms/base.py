from sentence_transformers import SentenceTransformer
from typing import List
from injector import singleton

class LLM:
    model_type = ''
    model_name = ''
    tokenizer = None
    llm = None

class Embeddings:
    model_name = ''
    embeddings = None
    using_custom_embedding_model = False

    def __init__(self, **kw) -> None:
        self.using_custom_embedding_model = kw.get('using_custom_embedding_model', False)
        if self.using_custom_embedding_model:
            self.embedding_model = kw.get('custom_embedding_model', None)
            if not self.embedding_model:
                raise ValueError('custom_embedding_model is None')
            self.custom_embeddings = CustomEmbeddings(self.embedding_model)

    def _encode(self, inputs: str, **kw):
        raise UnboundLocalError('没有实现encode方法')
    
    def encode(self, inputs: str, **kw):
        if self.using_custom_embedding_model:
            embeddings = self.encode_common(inputs, **kw)
        else:
            embeddings = self._encode(inputs, **kw)
        return embeddings
    
    def encode_common(self, inputs: str | List[str], **kw):
        # model = SentenceTransformer(self.custom_embedding_model)
        # embeddings = model.encode(inputs, normalize_embeddings=True)
        embeddings = self.custom_embeddings.encode(inputs, **kw)
        return embeddings
    

@singleton
class CustomEmbeddings(Embeddings):
    def __init__(self, embedding_model: str) -> None:
        self.model = SentenceTransformer(embedding_model, device='cuda', trust_remote_code=True)

    def encode(self, inputs: str | List[str], **kw):
        embeddings = self.model.encode(inputs, normalize_embeddings=True)
        return embeddings
