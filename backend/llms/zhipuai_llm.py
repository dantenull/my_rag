from .base import LLM, Embeddings

class ZhipuaiLLM(LLM):
    def __init__(self, engine: str, **kw) -> None:
        try:
            import os
            from zhipuai import ZhipuAI
        except ImportError as e:
            raise ImportError('pip install zhipuai') from e    
        openai_api_key = os.environ.get('ZHIPUAI_API_KEY')
        if not openai_api_key:
            raise ValueError('no ZHIPUAI_API_KEY')
    
        self.model_type = 'openai'
        self.model_name = 'openai'
        self.llm = ZhipuAI(api_key=openai_api_key)
        self.tokenizer = ZhipuaiEmbeddings(self.llm.embeddings, **kw)
        self.engine = engine if engine else 'glm-4'
    
    def chat(self, prompt: str) -> str:
        response = self.llm.chat.completions.create(
            model=self.engine,
            messages=[{'role': 'user', 'content': prompt}],
            # n=1,
            # temperature=0
        )
        return response.choices[0].message.content


class ZhipuaiEmbeddings(Embeddings):
    def __init__(self, embeddings, **kw) -> None:
        self.model_name = 'zhipuai'
        self.embeddings = embeddings
        self.embedding_model = kw.get('embedding_model', 'embedding-2')
        super().__init__(**kw)

    
    def _encode(self, inputs: str, **kw):
        # text = text.replace("\n", " ")
        # 直接传列表会报错，所以只能一个一个传
        return [self.embeddings.create(input=input, model=self.embedding_model).data[0].embedding for input in inputs]
