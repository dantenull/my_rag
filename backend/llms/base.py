
class LLM:
    model_type = ''
    model_name = ''
    tokenizer = None
    llm = None

class Embeddings:
    model_name = ''
    embeddings = None

    def encode(self, text: str, **kw):
        raise UnboundLocalError('没有实现encode方法')
