# from transformers import AutoModelForCausalLM, AutoTokenizer
from injector import inject, singleton
from settings.settings import Settings
from llms import LLM, OpenaiLLM, HuggingfaceLLM, ZhipuaiLLM


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
                openai_llm = OpenaiLLM(
                    engine, 
                    using_custom_embedding_model=settings.using_custom_embedding_model, 
                    custom_embedding_model=settings.custom_embedding_model
                )
                self.llm = openai_llm
                # self.tokenizer = openai_llm.tokenizer
                # self.model_name = openai_llm.model_name
            case 'zhipuai':
                engine = settings.zhipuai_model_engine
                zhipuai_llm = ZhipuaiLLM(
                    engine, 
                    using_custom_embedding_model=settings.using_custom_embedding_model, 
                    custom_embedding_model=settings.custom_embedding_model
                )
                self.llm = zhipuai_llm
            case 'mock':
                # TODO
                llm = LLM()
                self.llm = llm
                # self.tokenizer = llm.tokenizer
                # self.model_name = llm.model_name
       