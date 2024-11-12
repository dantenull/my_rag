# from transformers import AutoModelForCausalLM, AutoTokenizer
from injector import inject, singleton
from settings.settings import Settings
from llms import LLM, OpenaiLLM, LocalLLM, ZhipuaiLLM


@singleton
class LLMComponent:
    @inject
    def __init__(self, settings: Settings) -> None:
        # torch.manual_seed(0)
        self.settings = settings
        match settings.llm.mode:
            case 'local':
                model = settings.llm.model
                local_llm = LocalLLM(model)
                self.llm = local_llm
                # self.tokenizer = huggingface_llm.tokenizer
                # self.model_name = huggingface_llm.model_name
            case 'openai':
                model = settings.llm.model
                openai_llm = OpenaiLLM(
                    model, 
                    # custom_embedding_model=settings.embeddings.model,
                    api_base=settings.llm.api_base
                )
                self.llm = openai_llm
                # self.tokenizer = openai_llm.tokenizer
                # self.model_name = openai_llm.model_name
            case 'zhipuai':
                model = settings.llm.model
                zhipuai_llm = ZhipuaiLLM(
                    model, 
                    # custom_embedding_model=settings.embeddings.model
                )
                self.llm = zhipuai_llm
            case 'mock':
                # TODO
                llm = LLM()
                self.llm = llm
                # self.tokenizer = llm.tokenizer
                # self.model_name = llm.model_name
       