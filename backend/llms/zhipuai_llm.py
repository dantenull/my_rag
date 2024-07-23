from .base import LLM, Embeddings

class ZhipuaiLLM(LLM):
    def __init__(self, engine: str) -> None:
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
        self.tokenizer = ZhipuaiEmbeddings(self.llm.embeddings)
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
    def __init__(self, embeddings, embeddings_model: str = '') -> None:
        self.model_name = 'zhipuai'
        self.embeddings = embeddings
        self.embeddings_model = embeddings_model if embeddings_model else 'embedding-2'
    
    def encode(self, text: str, **kw):
        text = text.replace("\n", " ")
        return (
            self.embeddings.create(input=text, model=self.embeddings_model, **kw).data[0].embedding
        )
