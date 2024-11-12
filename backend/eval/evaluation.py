from component.llm.llm_component import LLMComponent
from component.embeddings.embeddings_component import EmbeddingsComponent
from settings.settings import Settings
from db.mongodb import MyMongodb
from db.es_client import ElasticsearchClient
from celery_app import process_async_insert_to_es, insert_to_milvus, insert_chunk_to_mongodb
from vectorstores.milvus import Milvus
# from query_optimizer import *
from tools import reciprocal_rank_fusion

class Evaluation:
    def __init__(
        self, 
        llm_component: LLMComponent, 
        embeddings_component: EmbeddingsComponent, 
        db: MyMongodb,
        es_client: ElasticsearchClient,
        settings: Settings, 
        # file_path: str,
        # eval_num: int=100,
    ) -> None:
        self.settings = settings
        self.llm = llm_component.llm
        self.embeddings = embeddings_component.embeddings
        self.vectorstore = Milvus(settings.milvus.uri, settings.milvus.port, settings.milvus.database)
        self.db = db
        self.es_client = es_client
        self.embeddings_model = settings.embeddings.model_name
