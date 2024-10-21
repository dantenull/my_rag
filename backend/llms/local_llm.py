from .base import LLM, Embeddings
from pathlib import Path

class LocalLLM(LLM):
    def __init__(self, model: str, **kw) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError('pip install transformers') from e   
        self.model_type = 'huggingface'
        self.model_name = Path(model).name
        # self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer = LocalEmbeddings(model, AutoTokenizer.from_pretrained(model), **kw)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model, 
            device_map='auto', 
            torch_dtype="auto",
            # trust_remote_code=True,
        )
    
    def chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "你是一个乐于助人的助手。"},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to('cuda')
        generated_ids = self.llm.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    

class LocalEmbeddings(Embeddings):
    def __init__(self, path: str, embeddings, **kw) -> None:
        self.model_name = Path(path).name
        self.embeddings = embeddings
    
    def encode(self, inputs: str, **kw):
        return self.embeddings.encode(inputs, **kw)