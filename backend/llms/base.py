from sentence_transformers import SentenceTransformer
from typing import List
from injector import inject, singleton
from settings.settings import Settings

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
            self.custom_embedding_model = kw.get('custom_embedding_model', None)
            if not self.custom_embedding_model:
                raise ValueError('custom_embedding_model is None')

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
        embeddings = CustomEmbeddings().encode(inputs, **kw)
        return embeddings
    

@singleton
class CustomEmbeddings(Embeddings):
    @inject
    def __init__(self, settings: Settings=None) -> None:
        # self.model_name = 'custom_embeddings'
        print(settings)
        self.custom_embedding_model = settings.custom_embedding_model
        self.model = SentenceTransformer(self.custom_embedding_model, device='cuda')

    def encode(self, inputs: str | List[str], **kw):
        embeddings = self.model.encode(inputs, normalize_embeddings=True)
        return embeddings
