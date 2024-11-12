from pydantic import BaseModel, Field
from typing import Literal
from settings.settings_loader import load_active_settings, PROJECT_ROOT_PATH
from .read_dotenv import read_dotenv


class LLMSettings(BaseModel):
    mode: Literal['openai', 'local', 'mock']
    model: str 
    api_base: str | None = None
    # size: int
    # device: Literal['auto', 'cuda', 'cpu']


class EmbeddingsSettings(BaseModel):
    # model: Optional[str] = Field(
    #     default=None,
    #     description='自定义embedding模型路径'
    # )
    # model_name: Optional[str] = Field(
    #     default=None,
    #     description='自定义embedding模型名称'
    # )
    mode: Literal['openai', 'local', 'mock']
    model: str | None = Field(
        # default=None,
        description='自定义embedding模型路径'
    )
    model_name: str | None = Field(
        # default=None,
        description='自定义embedding模型名称'
    )
    # using_custom_embedding_model: bool = Field(
    #     description='是否使用自定义embedding模型'
    # )
    dim: int
    api_base: str | None = None


class MongodbSettings(BaseModel):
    port: int
    db_name: str


class VectorstoreSettings(BaseModel):
    database: Literal['chroma', 'milvus']


class ChromaSettings(BaseModel):
    collection: str


class MilvusSettings(BaseModel):
    uri: str
    port: int
    database: str


class ElasticsearchSettings(BaseModel):
    host: str
    user: str


class RerankSettings(BaseModel):
    cross_encoder_path: str


class EntityExtractionSetting(BaseModel):
    model_path: str


class Settings(BaseModel):
    llm: LLMSettings
    embeddings: EmbeddingsSettings
    mongodb: MongodbSettings
    vectorstore: VectorstoreSettings
    chroma: ChromaSettings
    milvus: MilvusSettings
    elasticsearch: ElasticsearchSettings
    rerank: RerankSettings
    extraction: EntityExtractionSetting


# class Settings(BaseModel):
#     llm_mode: Literal['openai', 'huggingface', 'zhipuai', 'mock']
#     llm_model_path: str = Field(
#         # '/root/autodl-tmp/qwen1.5-7B-chat-sft',
#         # '/root/autodl-tmp/Qwen1.5-7B-Chat',                                                   
#         description='模型路径'
#     )
#     llm_size: int
#     custom_embedding_model: str = Field(
#         description='自定义embedding模型路径'
#     )
#     custom_embedding_model_name: str = Field(
#         description='自定义embedding模型名称'
#     )
#     using_custom_embedding_model: bool = Field(
#         description='是否使用自定义embedding模型'
#     )
#     device: Literal['auto', 'cuda', 'cpu']
#     mongodb_port: int
#     mongodb_db_name: str
#     cross_encoder_path: str
#     openai_model_engine: str
#     zhipuai_model_engine: str

#     chroma_collection: str

#     es_host: str
#     es_user: str
#     # es_password: str


read_dotenv(PROJECT_ROOT_PATH)

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
