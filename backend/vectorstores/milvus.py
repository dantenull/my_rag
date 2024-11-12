from pymilvus import connections, db, FieldSchema, CollectionSchema, DataType, Collection, MilvusClient
from typing import List, Dict, Union


class Milvus:
    def __init__(
        self, 
        uri: str = 'http://localhost', 
        port : int = 19530, 
        database: str = 'default',
        **kw
    ) -> None:
        self.client = MilvusClient(uri=uri, port=port, db_name=database, **kw)
        databases = self.client.list_databases()
        if database not in databases:
            self.client.create_database(database)
        self.client.using_database(database)
    
    def _get_schema(self, dim: int):
        id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True, description="primary id")
        embedding_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim, description="embedding")
        file_id_field = FieldSchema(name='file_id', dtype=DataType.VARCHAR, max_length=256, is_partition_key=True, description='file id')
        file_name_field = FieldSchema(name='file_name', dtype=DataType.VARCHAR, max_length=256, description='file name')
        model_name_field = FieldSchema(name='model_name', dtype=DataType.VARCHAR, max_length=256, description='model name')
        words_num_field = FieldSchema(name='words_num', dtype=DataType.INT32,  description='words num')
        document_field = FieldSchema(name='document', dtype=DataType.VARCHAR, max_length=65535, description='chunk with context')
        chunk_field = FieldSchema(name='chunk', dtype=DataType.VARCHAR, max_length=65535, description='chunk')
        seq_field = FieldSchema(name='seq', dtype=DataType.INT32, description='sequence')

        schema = CollectionSchema(
            fields=[id_field, embedding_field, file_id_field, file_name_field, model_name_field, words_num_field, document_field, chunk_field, seq_field], 
            auto_id=False, 
            enable_dynamic_field=True, 
            description=f"{dim} dim vector collection"
        )
        return schema

    def create_or_get_collection_vector(self, dim: int):
        # collection_name = 'vector_openai_ada002'
        collection_name = f'vector_{dim}'
        if self.client.has_collection(collection_name=collection_name):
            self.client.load_collection(
                collection_name=collection_name,
                replica_number=1
            )
        else:
            schema = self._get_schema(dim)
            index_params = [
                {
                    'field_name': 'embedding',
                    'index_name': "vector_index",
                    'metric_type': "COSINE",
                    'index_type': "HNSW",
                    'params': {
                        'M': 16,
                        'efConstruction': 200
                    }
                }
            ]
            self.client.create_collection(collection_name=collection_name, schema=schema, using='default', shards_num=2, index_params=index_params)
        return collection_name
    
    def insert_data(self, dim: int, data: Union[List[Dict], Dict], **kw):
        collection_name = self.create_or_get_collection_vector(dim)
        result = self.client.insert(collection_name, data, **kw)
        # return len(data) == result['insert_count']
        return result
    
    def search_data(self, dim: int, data: List, limit: int = 50, **kw) -> List[Dict]:
        collection_name = self.create_or_get_collection_vector(dim)
        result = self.client.search(collection_name, data=data, limit=limit, **kw)
        # print(result)
        return result[0] if result else []
    
    def get_data(self, dim: int, ids, **kw) -> List[Dict]:
        collection_name = self.create_or_get_collection_vector(dim)
        result = self.client.get(collection_name, ids=ids, **kw)
        return result

    def delete_data(
            self, 
            dim: int, 
            ids: List = None, 
            filter = '', 
            partition_name = '', 
            **kw
        ):
        collection_name = self.create_or_get_collection_vector(dim)
        result = self.client.delete(collection_name, ids=ids, filter=filter, partition_name=partition_name, **kw)
        return result
