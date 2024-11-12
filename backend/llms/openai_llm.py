from .base import LLM
from typing import List
from injector import singleton


@singleton
class OpenaiLLM(LLM):
    def __init__(self, model: str, **kw) -> None:
        try:
            import os
            from openai import OpenAI
        except ImportError as e:
            raise ImportError('pip install openai') from e    
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        # print('openai_api_key ', openai_api_key)
        if not openai_api_key:
            raise ValueError('no OPENAI_API_KEY')
    
        self.model_type = 'openai'
        self.model = 'openai'
        self.llm = OpenAI(api_key=openai_api_key, base_url=kw.get('api_base', None))
        # self.tokenizer = OpenaiEmbeddings(self.llm.embeddings, **kw)
        self.model_name = model if model else 'gpt-3.5-turbo'
    
    def chat(self, query: str, prompt: str = '') -> str:
        if prompt:
            messages = [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': query}
            ]
        else:
            messages = [
                {'role': 'user', 'content': query}
            ]
        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=1,
            temperature=0
        )
        return response.choices[0].message.content
    
    # def openai_tokenizer(self, text: str, **kwargs: Any) -> List[float]:
    #     text = text.replace("\n", " ")
    #     return (
    #         self.llm.embeddings.create(input=[text], model=self.engine, **kwargs).data[0].embedding
    #     )

# class OpenaiEmbeddings(Embeddings):
#     def __init__(self, embeddings, **kw) -> None:
#         self.model_name = 'openai'
#         self.embeddings = embeddings
#         # TODO 模型名加入settings
#         self.embedding_model = kw.get('embedding_model', 'text-embedding-ada-002')
#         super().__init__(**kw)
    
#     def _encode(self, inputs: str | List[str], **kw):
#         # text = text.replace("\n", " ")
#         data = self.embeddings.create(input=inputs, model=self.embedding_model).data
#         return [d.embedding for d in data]
        
