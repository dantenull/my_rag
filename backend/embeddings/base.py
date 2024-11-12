from sentence_transformers import SentenceTransformer
from typing import List, Union
from injector import singleton

class Embeddings:
    embeddings_type = ''
    embeddings_name = ''
    embeddings = None
    using_custom_embedding_model = False

    # def __init__(self, **kw) -> None:
    #     # self.using_custom_embedding_model = kw.get('using_custom_embedding_model', False)
    #     custom_embedding_model = kw.get('custom_embedding_model', None)
    #     self.using_custom_embedding_model = (custom_embedding_model != '' and custom_embedding_model is not None)
    #     if self.using_custom_embedding_model:
    #         self.embedding_model = custom_embedding_model
    #         if not self.embedding_model:
    #             raise ValueError('custom_embedding_model is None')
    #         self.custom_embeddings = CustomEmbeddings(self.embedding_model)

    def encode(self, inputs: Union[str, List], **kw):
        raise UnboundLocalError('没有实现encode方法')
    
    # def encode(self, inputs: str, **kw):
    #     if self.using_custom_embedding_model:
    #         embeddings = self.encode_common(inputs, **kw)
    #     else:
    #         embeddings = self._encode(inputs, **kw)
    #     return embeddings
    
    # def encode_common(self, inputs: str | List[str], **kw):
    #     # model = SentenceTransformer(self.embedding.model)
    #     # embeddings = model.encode(inputs, normalize_embeddings=True)
    #     embeddings = self.custom_embeddings.encode(inputs, **kw)
    #     return embeddings
