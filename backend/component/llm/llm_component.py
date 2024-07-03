# from transformers import AutoModelForCausalLM, AutoTokenizer
from injector import inject, singleton
from settings.settings import Settings
from llms import LLM, OpenaiLLM, HuggingfaceLLM


@singleton
class LLMComponent:
    @inject
    def __init__(self, settings: Settings) -> None:
        # torch.manual_seed(0)
        self.settings = settings
        match settings.llm_mode:
            case 'huggingface':
                path = settings.llm_model_path
                huggingface_llm = HuggingfaceLLM(path)
                self.llm = huggingface_llm
                # self.tokenizer = huggingface_llm.tokenizer
                # self.model_name = huggingface_llm.model_name
            case 'openai':
                engine = settings.openai_model_engine
                openai_llm = OpenaiLLM(engine)
                self.llm = openai_llm
                # self.tokenizer = openai_llm.tokenizer
                # self.model_name = openai_llm.model_name
            case 'mock':
                # TODO
                llm = LLM()
                self.llm = llm
                # self.tokenizer = llm.tokenizer
                # self.model_name = llm.model_name
       