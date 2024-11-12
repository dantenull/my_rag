from sentence_transformers import SentenceTransformer
from typing import List
from injector import singleton
from openai.resources import Embeddings as openai_embeddings

class LLM:
    model = ''  # 模型名称或路径。如openai或/path/to/model
    model_type = ''  # 模型类别。如openai，local
    model_name = ''  # 模型具体名称。
    # tokenizer = None
    llm = None

# class Embeddings:
#     model_name = ''
#     embeddings = None
#     using_custom_embedding_model = False

#     def __init__(self, **kw) -> None:
#         # self.using_custom_embedding_model = kw.get('using_custom_embedding_model', False)
#         custom_embedding_model = kw.get('custom_embedding_model', None)
#         self.using_custom_embedding_model = (custom_embedding_model != '' and custom_embedding_model is not None)
#         if self.using_custom_embedding_model:
#             self.embedding_model = custom_embedding_model
#             if not self.embedding_model:
#                 raise ValueError('custom_embedding_model is None')
#             self.custom_embeddings = CustomEmbeddings(self.embedding_model)

#     def _encode(self, inputs: str, **kw):
#         raise UnboundLocalError('没有实现encode方法')
    
#     def encode(self, inputs: str, **kw):
#         if self.using_custom_embedding_model:
#             embeddings = self.encode_common(inputs, **kw)
#         else:
#             embeddings = self._encode(inputs, **kw)
#         return embeddings
    
#     def encode_common(self, inputs: str | List[str], **kw):
#         # model = SentenceTransformer(self.embedding.model)
#         # embeddings = model.encode(inputs, normalize_embeddings=True)
#         embeddings = self.custom_embeddings.encode(inputs, **kw)
#         return embeddings
    

# @singleton
# class CustomEmbeddings:
#     def __init__(self, embedding_model: str) -> None:
#         self.model = SentenceTransformer(embedding_model, device='cuda', trust_remote_code=True)

#     def encode(self, inputs: str | List[str], **kw):
#         embeddings = self.model.encode(inputs, normalize_embeddings=True)
#         return embeddings
