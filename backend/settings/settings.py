from pydantic import BaseModel, Field
from typing import Literal
from settings.settings_loader import load_active_settings

class Settings(BaseModel):
    llm_mode: Literal['openai', 'huggingface', 'zhipuai', 'mock']
    llm_model_path: str = Field(
        # '/root/autodl-tmp/qwen1.5-7B-chat-sft',
        # '/root/autodl-tmp/Qwen1.5-7B-Chat',
        description='模型路径'
    )
    llm_size: int = Field()
    device: Literal['auto', 'cuda', 'cpu'] = Field()
    mongodb_port: int= Field()
    mongodb_db_name: str = Field()
    cross_encoder_path: str = Field()
    openai_model_engine: str
    zhipuai_model_engine: str

    chroma_collection: str

    es_host: str
    es_user: str
    # es_password: str

"""
This is visible just for DI or testing purposes.

Use dependency injection or `settings()` method instead.
"""
unsafe_settings = load_active_settings()

"""
This is visible just for DI or testing purposes.

Use dependency injection or `settings()` method instead.
"""
unsafe_typed_settings = Settings(**unsafe_settings)
