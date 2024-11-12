from .base import Embeddings
from typing import List
from injector import singleton
from openai import OpenAI
import time


@singleton
class OpenaiEmbeddings(Embeddings):
    def __init__(self, embeddings_name: str, api_base: str, **kw) -> None:
        self.embeddings_type = 'openai'
        self.embeddings_name = embeddings_name
        # self.embeddings = embeddings
        self.embeddings = OpenAI(base_url=api_base).embeddings
        self.api_base = api_base
    
    def encode(self, inputs: str | List[str], **kw):
        # text = text.replace("\n", " ")
        index = 0
        # loop = True
        while True:
            try:
                data = self.embeddings.create(input=inputs, model=self.embeddings_name, **kw).data
                return [d.embedding for d in data]
            except Exception as e:
                print(str(e))
                index += 1
                if index > 12:
                    return []
                time.sleep(5)
                
