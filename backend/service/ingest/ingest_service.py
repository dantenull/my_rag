from injector import inject, singleton
from settings.settings import Settings
from component.llm.llm_component import LLMComponent
from pathlib import Path
from component.ingest.ingest_component import IngestComponent
from db.mongodb import MyMongodb
from db.es_client import ElasticsearchClient
from typing import List, Dict

@singleton
class IngestService:
    @inject
    def __init__(self, llm_component: LLMComponent, db: MyMongodb, setting: Settings, es_client: ElasticsearchClient) -> None:
        self.llm = llm_component.llm
        self.db = db
        self.embedding_model = self.llm.tokenizer
        self.ingest_component = IngestComponent(llm_component, self.db, setting, es_client)
    
    def ingest_file(self, file_name: str, fileb: bytes, **kw):
        return self.ingest_component.ingest_file(file_name, fileb, **kw)
    
    def ingest_file_local(self, file_path: str):
        return self.ingest_component.ingest_file_local(file_path)

    # def ingest_file_by_semantic(self, file_path: str):
    #     self.ingest_component.ingest_file_by_semantic(file_path)
    
    def list_ingested(self) -> List[Dict]:
        return self.ingest_component.file_list()
    
    def get_documents(self, file_name: str, pages_index: list[int]):
        return self.ingest_component.get_documents(file_name, pages_index)
    
    def get_file_info(self, file_name: str):
        return self.ingest_component.get_file_info(file_name)
    
    async def delete_by_file(self, file_name: str):
        await self.ingest_component.delete_by_file(file_name)
    
    # def get_celery_task_status(self, task_id: str):
    #     return self.ingest_component.get_celery_task_status(task_id)
    