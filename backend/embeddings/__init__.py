from .base import Embeddings
from .openai_embeddings import OpenaiEmbeddings
from .local_embeddings import LocalEmbeddings

__all__ = ['Embeddings', 'OpenaiEmbeddings', 'LocalEmbeddings']
