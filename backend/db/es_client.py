from elasticsearch import Elasticsearch, helpers
from elasticsearch import AsyncElasticsearch
from injector import inject, singleton
from typing import List, Dict, Iterable, Any
from schema import Document
from settings.settings import Settings
import os

class ElasticsearchClientBase:
    def __init__(self, es_host: str, es_user: str, es_password: str) -> None:
        self.connects = {
            'hosts': es_host,
            'basic_auth': (es_user, es_password),
        }
        print(self.connects)
        self.client = AsyncElasticsearch(
            **self.connects, 
            verify_certs=False,
            ca_certs=None, 
            # client_cert=None,
            # client_key=None,
            # ssl_assert_hostname=None,
            # ssl_assert_fingerprint=None,
            # ssl_context=None,
        )
        self.check_connect()
    
    async def check_connect(self):
        if await self.client.ping():
            print("Elasticsearch 连接成功！")
        else:
            print("无法连接到 Elasticsearch 服务。")
        self.client_info = await self.client.info()
        print(self.client_info)
    
    async def create_index(self, index_name: str) -> None:
        is_exists = await self.client.indices.exists(index=index_name)
        if is_exists:
            pass
        else:
            settings = {
                "index": {
                    "similarity": {
                        "custom_bm25": {
                            "type": "BM25",
                            "k1": "1.3",
                            "b": "0.6"
                        }
                    }
                }
            }
            mappings = {
                "properties": {
                    'doc_id': {
                        'type': 'keyword',
                        'index': True
                    },
                    "text": {
                        "type": "text",
                        "similarity": "custom_bm25",
                        "index": True,
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_smart",
                    }
                }
            }
            self.client.indices.create(index=index_name, mappings=mappings, settings=settings)

    async def async_insert(self, index_name: str, data: List[Dict]) -> None:
        await self.create_index(index_name)
        actions = []
        for item in data:
            action = {
                '_op_type': 'index',
                '_id': item['doc_id'],
                'doc_id': item['doc_id'],
                'text': item['text'],
                **item['metadata']
            }
            actions.append(action)
        
        documents_written_count, errors = await helpers.async_bulk(
            client=self.client,
            actions=actions,
            refresh=False,
            index=index_name,
            stats_only=True,
            raise_on_error=False,
        )
        print(documents_written_count, errors)
        return documents_written_count, errors
    
    async def async_search_docs(self, index_name: str, query) -> Iterable[Dict[str, Any]]:
        result = []
        async for value in helpers.async_scan(
            client=self.client,
            query=query,
            index=index_name,
        ):
            result.append(value)
        return result

    async def async_search_file(self, index_name: str, file_name: str) -> Iterable[Dict[str, Any]]:
        return await self.async_search_docs(index_name, {'query': {'match': {'file_name': file_name}}})

    async def async_delete_file(self, index_name: str, file_name: str) -> None:
        data = await self.async_search_file(index_name, file_name)
        try:
            ids = [d['_id'] for d in data]
            await self.async_delete(index_name, ids)
        except Exception as e:
            # TODO
            print('async_delete_file\n' + e)

    async def async_delete(self, index_name: str, ids: List[int]):
        actions = []
        for i in ids:
            action = {
                '_op_type': 'delete',
                '_index': index_name,
                '_id': i,
            }
            actions.append(action)
        documents_written_count, errors = await helpers.async_bulk(
            client=self.client,
            actions=actions
        )
        print(documents_written_count, errors)
    
    def insert(self, index_name: str, data: List[Document]) -> None:
        self.create_index(index_name)
        actions = []
        for item in data:
            action = {
                '_op_type': 'index',
                '_id': item.doc_id,
                'doc_id': item.doc_id,
                'text': item.text,
                **item.metadata
            }
            actions.append(action)
        
        documents_written_count, errors = helpers.bulk(
            client=self.client,
            actions=actions,
            refresh=False,
            index=index_name,
            stats_only=True,
            raise_on_error=False,
        )
        print(documents_written_count, errors)
    
    def search_docs(self, index_name: str, query) -> Iterable[Dict[str, Any]]:
        return helpers.scan(
            client=self.client,
            query=query,
            index=index_name,
        )

    def search_file(self, index_name: str, file_name: str) -> Iterable[Dict[str, Any]]:
        return self.search_docs(index_name, {'query': {'match': {'file_name': file_name}}})

    def delete_file(self, index_name: str, file_name: str) -> None:
        data = self.search_file(index_name, file_name)
        try:
            ids = [d['_id'] for d in data]
            self.delete(index_name, ids)
        except:
            # TODO
            pass

    def delete(self, index_name: str, ids: List[int]):
        actions = []
        for i in ids:
            action = {
                '_op_type': 'delete',
                '_index': index_name,
                '_id': i,
            }
            actions.append(action)
        documents_written_count, errors = helpers.bulk(
            client=self.client,
            actions=actions
        )
        print(documents_written_count, errors)

@singleton
class ElasticsearchClient(ElasticsearchClientBase):
    @inject
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.es_host, settings.es_user, os.getenv('ELASTIC_PASSWORD'))
