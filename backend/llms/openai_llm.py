from .base import LLM, Embeddings
from typing import Any,  List 

class OpenaiLLM(LLM):
    def __init__(self, engine: str) -> None:
        try:
            import os
            from openai import OpenAI
        except ImportError as e:
            raise ImportError('pip install openai') from e    
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError('no OPENAI_API_KEY')
    
        self.model_type = 'openai'
        self.model_name = 'openai'
        self.llm = OpenAI(api_key=openai_api_key)
        self.tokenizer = OpenaiEmbeddings(self.llm.embeddings)
        self.engine = engine if engine else 'gpt-3.5-turbo'
    
    def chat(self, prompt: str) -> str:
        response = self.llm.chat.completions.create(
            model=self.engine,
            messages=[{'role': 'user', 'content': prompt}],
            n=1,
            temperature=0
        )
        return response.choices[0].message.content
    
    # def openai_tokenizer(self, text: str, **kwargs: Any) -> List[float]:
    #     text = text.replace("\n", " ")
    #     return (
    #         self.llm.embeddings.create(input=[text], model=self.engine, **kwargs).data[0].embedding
    #     )

class OpenaiEmbeddings(Embeddings):
    def __init__(self, embeddings, embeddings_model: str = '') -> None:
        self.model_name = 'openai'
        self.embeddings = embeddings
        self.embeddings_model = embeddings_model if embeddings_model else 'text-embedding-3-small'
    
    def encode(self, text: str, **kw):
        text = text.replace("\n", " ")
        return (
            self.embeddings.create(input=[text], model=self.embeddings_model, **kw).data[0].embedding
        )