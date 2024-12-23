'''
celery -A celery_app worker --loglevel=info -P eventlet --concurrency=10   
celery -A celery_app flower --address=127.0.0.1 --port=5566
'''

from celery import Celery
from typing import List, Dict, Iterable, Union, Any
from db.es_client import ElasticsearchClientBase
from db.mongodb import MyMongodbBase
from settings.settings import unsafe_typed_settings as rag_settings
import os
import asyncio
from celery.utils.log import get_task_logger
# from celery.schedules import crontab
import chromadb
from chromadb import Documents, EmbeddingFunction
from openai import OpenAI, AsyncOpenAI
from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer
import traceback
from tools import get_chunk_context
from celery.signals import task_prerun, task_postrun, task_success, task_failure
import time
from datetime import datetime
from vectorstores.milvus import Milvus
from component.embeddings.embeddings_component import EmbeddingsComponent
from extraction.extraction import ExtractionBase 
from component.llm.llm_component import LLMComponent
import networkx as nx
from collections import defaultdict
import uuid

# class CustomEmbeddings:
#     def __init__(self) -> None:
#         self.model = SentenceTransformer(rag_settings.embeddings.model, device='cuda', trust_remote_code=True)

#     def encode(self, inputs: str | List[str], **kw):
#         embeddings = self.model.encode(inputs, normalize_embeddings=True)
#         return embeddings


# custom_embeddings = CustomEmbeddings()

openai_api_key = 'sk-Ie9c0c27f1a1444b286ee5efa4e1e6db920a08f193e5ov9A'
openai_llm = OpenAI(api_key=openai_api_key, base_url=rag_settings.llm.api_base)

# class MyEmbeddingFunction(EmbeddingFunction):
#     def __init__(self) -> None:
#         self.llm = None
#         self.llm_mode = rag_settings.llm.mode
#         if self.llm_mode == 'openai':
#             # openai_api_key = os.environ.get('OPENAI_API_KEY')
#             openai_api_key = 'sk-Ie9c0c27f1a1444b286ee5efa4e1e6db920a08f193e5ov9A'
#             self.llm = OpenAI(api_key=openai_api_key, base_url=rag_settings.llm.api_base)
#             self.embedding_model = 'text-embedding-ada-002'
#         elif self.llm_mode == 'zhipuai':
#             zhipuai_api_key = os.environ.get('ZHIPUAI_API_KEY')
#             self.llm = ZhipuAI(api_key=zhipuai_api_key)
#             self.embedding_model = 'embedding-2'
#         if self.llm:
#             self.embeddings = self.llm.embeddings
#         super().__init__()

#     def __call__(self, input: Documents) -> List[List[float]]:
#         if not self.llm:
#             return []
#         embeddings = []
#         input = [text.replace("\n", " ") for text in input if text]
#         # print([len(text) for text in input if len(text) > 512])
#         embeddings = self.encode(input)
#         return [list(map(float, e)) for e in embeddings]
    
#     def encode(self, inputs: List[str]):
#         if rag_settings.embeddings.model != '' and rag_settings.embeddings.model is not None:
#             return custom_embeddings.encode(inputs)
#         else:
#             return self._encode(inputs)
    
#     # def encode_common(self, inputs: str | List[str], **kw):
#     #     model = SentenceTransformer(rag_settings.embeddings.model)
#     #     embeddings = model.encode(inputs, normalize_embeddings=True)
#     #     return embeddings
    
#     def _encode(self, inputs: List[str]):
#         if self.llm_mode == 'openai':
#             return self._openai_encode(inputs)
#         elif self.llm_mode == 'zhipuai':
#             return self._zhipuai_encode(inputs)
    
#     def _openai_encode(self, inputs: List[str]):
#         data = self.embeddings.create(input=inputs, model=self.embedding_model).data
#         return [d.embedding for d in data]
    
#     def _zhipuai_encode(self, inputs: List[str]):
#         return [self.embeddings.create(input=input, model=self.embedding_model).data[0].embedding for input in inputs]
    
#     # def _zhipuai_encode_batch(self, input: List[str]):
#     #     input_file_id = 'file_123'
#     #     output_file_id = None
#     #     self.llm.batches.create(
#     #         input_file_id="file_123",
#     #         endpoint="/v4/embeddings",
#     #         completion_window="24h",
#     #     )
#     #     completed = False
#     #     while not completed:
#     #         retrieve = self.llm.batches.retrieve(input_file_id)
#     #         if retrieve['status'] != 'completed':
#     #             continue
#     #         output_file_id = retrieve['output_file_id']
#     #         content = self.llm.files.content(output_file_id) 


DEFAULT_RETRY_DELAY = 30
logger = get_task_logger(__name__)
celery_app = Celery(
    'tasks', 
    broker='redis://localhost:6379/0', 
    backend='redis://localhost:6379/0', 
)
celery_app.config_from_object('celeryconfig')
# celery_app.conf.beat_schedule = {
#     'retry-failed-tasks-every-5-minutes': {
#         'task': 'retry_failed_tasks',
#         'schedule': crontab(minute='*/1'),
#     },
# }
es_client = ElasticsearchClientBase(rag_settings.elasticsearch.host, rag_settings.elasticsearch.user, os.getenv('MYRAG_ELASTIC_PASSWORD'))
mongodb_client = MyMongodbBase(rag_settings.mongodb.port, rag_settings.mongodb.db_name, rag_settings.mongodb.username)
chroma_client = chromadb.PersistentClient(path='.\\chroma_db_test' + '_' + rag_settings.llm.model)
# chroma_client_collection = chroma_client.get_or_create_collection(rag_settings.chroma.collection, embedding_function=MyEmbeddingFunction())
milvus_client = Milvus(rag_settings.milvus.uri, rag_settings.milvus.port, rag_settings.milvus.database)
llm = LLMComponent(rag_settings).llm


def chat(query: str, prompt: str = '', history: List = []) -> str:
    if prompt:
        messages = [
            *history,
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query}
        ]
    else:
        messages = [
            *history,
            {'role': 'user', 'content': query}
        ]
    response = openai_llm.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        n=1,
        temperature=0
    )
    return response.choices[0].message.content


def get_celery_task_status(task_id: str):
    if not task_id:
        return {'status': 'Completed'}
    task_result = celery_app.AsyncResult(task_id)
    if task_result.state == 'SUCCESS':
        return {'status': 'Completed', 'result': task_result.result}
    elif task_result.state == 'PENDING':
        return {'status': 'Pending'}
    else:
        return {'status': task_result.state}

async def async_insert_to_es(file_id: str, index_name: str, data: List[Dict]):
    result = await es_client.async_insert(index_name, data)
    return result

'''
当前es任务中协程使用的loop。
因为es库在每次请求时会创建session对象，session中会初始化loop，
再次请求时直接使用这个session，意味着其中的loop不会更新。
这时如果重新创建loop会报错。
'''
current_es_task_loop = asyncio.new_event_loop()
current_mongodb_task_loop = None

def get_or_create_event_loop():
    global current_es_task_loop
    try:
        if not current_es_task_loop:
            loop = asyncio.get_event_loop()
            current_es_task_loop = loop
        else:
            loop = current_es_task_loop
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        logger.warning('get_or_create_event_loop RuntimeError')
        traceback.print_exc()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        current_es_task_loop = loop
        return loop

@celery_app.task(bind=True, default_retry_delay=0)
def test_task(self):
    pass


# def update_task_info(task_id: str, file_id: str):
#     result = mongodb_client.get_one_data('task_info', {'task_id': task_id})
#     if result:
#         mongodb_client.update_data('task_info', {
#             'file_id': file_id,
#         }, {'task_id': task_id})

@celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
def process_async_insert_to_es(self, file_id: str, index_name: str, data: List[Dict], **kw):
    # 写elasticsearch任务
    # TODO 去掉异步
    global current_es_task_loop
    try:
        # loop = get_or_create_event_loop()
        # if not current_es_task_loop:
        #     loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(loop)
        #     current_es_task_loop = loop
        # else:
        #     loop = current_es_task_loop
        # loop = asyncio.get_running_loop()
        get_or_create_event_loop()
        async_result = current_es_task_loop.run_until_complete(async_insert_to_es(file_id, index_name, data))
        # TODO 把以下两句放到callback里
        mongodb_client.update_fileinfo(file_id, {'upload_state_elasticsearch': 'done'})
        mongodb_client.update_upload_state(file_id)
        return async_result
    except RuntimeError as exc:
        logger.error(f"Task failed: {exc}, retrying...")
        traceback.print_exc()

        # 如果报RuntimeError的错，代表session可能过期（没有证实，属猜测），或是其他的问题，
        # 这时重新创建loop给current_es_task_loop赋值。
        # loop = get_or_create_event_loop()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        current_es_task_loop = loop

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, queue='failed')
        else:
            logger.error(f"Task failed after {self.max_retries} retries: {exc}")
            # self.apply_async(args=[file_name, index_name, data], queue='failed')
            raise
    except Exception as exc:
        logger.error(f"Task failed: {exc}, retrying...")
        traceback.print_exc()
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, queue='failed')
        else:
            logger.error(f"Task failed after {self.max_retries} retries: {exc}")
            # self.apply_async(args=[file_name, index_name, data], queue='failed')
            raise

@celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
def insert_to_mongodb(self, contents: List[Dict], file_name: str):
    # 写mongodb任务
    try:
        if contents and file_name:
            mongodb_client.insert_contents(contents, file_name)
        # TODO 把以下两句放到callback里
        mongodb_client.update_fileinfo(file_name, {'upload_state_no_sql': 'done'})
        mongodb_client.update_upload_state(file_name)
        return True
    except Exception as exc:
        logger.error(f"Task failed: {exc}, retrying...")
        traceback.print_exc()
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        else:
            logger.error(f"Task failed after {self.max_retries} retries: {exc}")
            # self.apply_async(args=[contents, file_name], queue='failed')
            raise

async def insert_chunk(chunk_info: Dict):
    chunk_context = await get_chunk_context(chat, chunk_info['chunk'], chunk_info['document'])
    chunk_info['context'] = chunk_context
    mongodb_client.insert_chunk(chunk_info)

async def insert_chunks(chunk_infos: List[Dict]):
    tasks = [asyncio.create_task(insert_chunk(chunk_info)) for chunk_info in chunk_infos]
    await asyncio.gather(*tasks)

@celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
def insert_chunks_to_mongodb(self, chunk_infos: List[Dict], file_id: str = '', **kw):
    global current_es_task_loop
    try:
        # if not current_es_task_loop:
        #     loop = asyncio.new_event_loop()
        #     asyncio.set_event_loop(loop)
        #     current_es_task_loop = loop
        # else:
        #     loop = current_es_task_loop
        get_or_create_event_loop()
        # TODO 不使用run_until_complete，改成同步
        async_result = current_es_task_loop.run_until_complete(insert_chunks(chunk_infos))
        # chunk_context = get_chunk_context(chat, chunk_info['chunk'], chunk_info['document'])
        # chunk_info['context'] = chunk_context
        # mongodb_client.insert_chunk(chunk_info)
    except RuntimeError as exc:
        logger.error(f"Task failed: {exc}, retrying...")
        traceback.print_exc()

        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # current_es_task_loop = loop

        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, queue='failed')
        else:
            logger.error(f"Task failed after {self.max_retries} retries: {exc}")
            # self.apply_async(args=[file_name, index_name, data], queue='failed')
            raise
    except Exception as exc:
        logger.error(f"Task failed: {exc}, retrying...")
        traceback.print_exc()
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        else:
            logger.error(f"Task failed after {self.max_retries} retries: {exc}")
            # self.apply_async(args=[contents, file_name], queue='failed')
            raise

@celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
def insert_chunk_to_mongodb(self, chunk_info: Dict, **kw):
    try:
        chunk_context = get_chunk_context(chat, chunk_info['chunk'], chunk_info['document'])
        chunk_info['context'] = chunk_context
        mongodb_client.insert_chunk(chunk_info)
    except Exception as exc:
        logger.error(f"Task failed: {exc}, retrying...")
        traceback.print_exc()
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        else:
            logger.error(f"Task failed after {self.max_retries} retries: {exc}")
            # self.apply_async(args=[contents, file_name], queue='failed')
            raise

@celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
def insert_to_milvus(self, file_id: str, **kw):
    try:
        chunks = mongodb_client.get_chunks(file_id)
        embeddings = EmbeddingsComponent(rag_settings).embeddings
        documents = [chunk['document'] for chunk in chunks]
        vectors = embeddings.encode(documents)
        # logger.info(str([len(v) for v in vectors]))
        # data = [{
        #     'id': chunk['id'],
        #     'embedding': vectors[i],
        #     'file_id': file_id,
        #     'file_name': chunk['metadata']['file_name'],
        #     'model_name': chunk['metadata']['embeddings_model'],
        #     'words_num': len(chunk),
        #     'document': chunk['document'],
        #     'chunk': chunk['chunk'],
        #     'seq': chunk['metadata']['seq'],
        # } for i, chunk in enumerate(chunks)]
        data = []
        for i, chunk in enumerate(chunks):
            d = {
                'id': chunk['id'],
                'embedding': vectors[i],
                'file_id': file_id,
                'file_name': chunk['metadata']['file_name'],
                'model_name': chunk['metadata']['embeddings_model'],
                'words_num': len(chunk),
                'document': chunk['document'],
                'chunk': chunk['chunk'],
                'seq': chunk['metadata']['seq'],
            }
            # del chunk['metadata']['file_name']
            # del chunk['metadata']['embeddings_model']
            # del chunk['metadata']['seq']
            for m in chunk['metadata']:
                if m in ['file_name', 'embeddings_model', 'seq']:
                    continue
                d[m] = chunk['metadata'][m]
            # d.update(chunk['metadata'])
            # print(d)
            # print(chunk['metadata'])
            data.append(d)
        milvus_client.insert_data(rag_settings.embeddings.dim, data, **kw)
        
        mongodb_client.update_fileinfo(file_id, {'upload_state_vectorstore': 'done'})
        mongodb_client.update_upload_state(file_id)
    except Exception as exc:
        logger.error(f"Task failed: {exc}, retrying...")
        # traceback.print_exc()
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        else:
            logger.error(f"Task failed after {self.max_retries} retries: {exc}")
            # self.apply_async(args=[contents, file_name], queue='failed')
            raise

@celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
def insert_graph_nodes_edges(self, file_id: str, **kw):
    try:
        extractor = ExtractionBase(rag_settings.extraction.mode, llm=llm)
        chunks = mongodb_client.get_chunks(file_id)
        # entities: Any, graph_data: nx.Graph = extractor.extract_graph(chunks)
        r = extractor.extract_graph(chunks)
        graph_data: nx.Graph = r[1]
        '''
        TODO 处理graph_data。
        整体信息入mongodb，desc入milvus，这两条数据做关联。
        最后在所有的实体抽取完了之后，在chunk库中标记每个chunk和实体关系的关联。 ok
        聚类clustering。
        其他：
        各个库之间字段关系图。
        '''
        vectors_data = []
        descriptions = []
        for name, node in graph_data.nodes(data=True):
            description = node['description']
            entity_id = str(uuid.uuid4())
            mongodb_client.insert_data('entities', {
                'id': entity_id,
                'name': name,
                'type': node['type'],
                'description': description,
                'source_ids': node['source_id'].split(', '),
                'file_id': file_id,
                # 'datasets': datasets_name
            })
            if not description:
                continue
            descriptions.append(description)
            vectors_data.append({
                'id': str(uuid.uuid4()),
                'file_id': file_id,
                'words_num': len(description),
                'document': description,
                'entity_id': entity_id,
                'file_name': '',
                'model_name': '',
                'chunk': '',
                'seq': -1,
            })
        
        for source, target, edge in graph_data.edges(data=True):
            description = edge['description']
            edge_id = str(uuid.uuid4())
            mongodb_client.insert_data('relationships', {
                'id': edge_id,
                'source': source,
                'target': target,
                'weight': edge['weight'],
                'description': description,
                'source_ids': edge['source_id'].split(', '),
                'file_id': file_id,
                # 'datasets': datasets_name
            })
            if not description:
                continue
            descriptions.append(description)
            vectors_data.append({
                'id': str(uuid.uuid4()),
                'file_id': file_id,
                'words_num': len(description),
                'document': description,
                'edge_id': edge_id,
                'file_name': '',
                'model_name': '',
                'chunk': '',
                'seq': -1,
            })
        
        embeddings = EmbeddingsComponent(rag_settings).embeddings
        vectors = embeddings.encode(descriptions)
        for i, vector in enumerate(vectors):
            vectors_data[i]['embedding'] = vector
        # for i, v in enumerate(vectors_data):
        #     print(f'index:{i}')
        #     print({
        #         'file_id': v['file_id'],
        #         'words_num': v['words_num'],
        #         'document': v['document'],
        #         'the_id': v['edge_id'] if 'edge_id' in v else v['entity_id'],
        #         'embedding': len(v['embedding']),
        #         'file_name': '',
        #         'model_name': '',
        #         'chunk': '',
        #         'seq': -1,
        #     })
        milvus_client.insert_data(rag_settings.embeddings.dim, vectors_data)
    except Exception as exc:
        logger.error(f"Task failed: {exc}, retrying...")
        # traceback.print_exc()
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        else:
            logger.error(f"Task failed after {self.max_retries} retries: {exc}")
            # self.apply_async(args=[contents, file_name], queue='failed')
            raise
    
@celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
def insert_graph_nodes_edges_descriptions(self, file_id: str, **kw):    
    try:
        vectors_data = []
        descriptions = []
        embeddings = EmbeddingsComponent(rag_settings).embeddings
        entities = mongodb_client.get_data('entities', {'file_id': file_id})
        for entity in entities:
            description = entity['description']
            if not description:
                continue
            entity_id = entity['id']
            descriptions.append(description)
            vectors_data.append({
                'id': str(uuid.uuid4()),
                'file_id': file_id,
                'words_num': len(description),
                'document': description,
                'entity_id': entity_id,
                'file_name': '',
                'model_name': '',
                'chunk': '',
                'seq': -1,
            })
        
        relationships = mongodb_client.get_data('relationships', {'file_id': file_id})
        for relationship in relationships:
            description = relationship['description']
            if not description:
                continue
            edge_id = relationship['id']
            descriptions.append(description)
            vectors_data.append({
                'id': str(uuid.uuid4()),
                'file_id': file_id,
                'words_num': len(description),
                'document': description,
                'edge_id': edge_id,
                'file_name': '',
                'model_name': '',
                'chunk': '',
                'seq': -1,
            })
        
        embeddings = EmbeddingsComponent(rag_settings).embeddings
        vectors = embeddings.encode(descriptions)
        for i, vector in enumerate(vectors):
            vectors_data[i]['embedding'] = vector
        # for i, v in enumerate(vectors_data):
        #     print(f'index:{i}')
        #     print({
        #         'file_id': v['file_id'],
        #         'words_num': v['words_num'],
        #         'document': v['document'],
        #         'the_id': v['edge_id'] if 'edge_id' in v else v['entity_id'],
        #         'embedding': len(v['embedding']),
        #         'embedding_type': type(v['embedding']),
        #         'file_name': v['file_name'],
        #         'model_name': v['model_name'],
        #         'chunk': v['chunk'],
        #         'seq': v['seq'],
        #     })
        milvus_client.insert_data(rag_settings.embeddings.dim, vectors_data)
    except Exception as exc:
        logger.error(f"Task failed: {exc}, retrying...")
        # traceback.print_exc()
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        else:
            logger.error(f"Task failed after {self.max_retries} retries: {exc}")
            # self.apply_async(args=[contents, file_name], queue='failed')
            raise

@celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
def insert_graph_chunk_fields(self, file_id: str):
    try:
        entities = mongodb_client.get_data('entities', {'file_id': file_id})
        entities_ids = defaultdict(list)
        for entity in entities:
            source_ids = entity['source_ids']
            for source_id in source_ids:
                entities_ids[source_id].append(entity['id'])
        
        relationships = mongodb_client.get_data('relationships', {'file_id': file_id})
        relationship_ids = defaultdict(list)
        for relationship in relationships:
            source_ids = relationship['source_ids']
            for source_id in source_ids:
                relationship_ids[source_id].append(relationship['id'])
        
        for source_id, ids in entities_ids:
            mongodb_client.update_data('chunks', {'id': source_id}, {'entities': ids})

        for source_id, ids in relationship_ids:
            mongodb_client.update_data('chunks', {'id': source_id}, {'relationships': ids})
    except Exception as exc:
        logger.error(f"Task failed: {exc}, retrying...")
        # traceback.print_exc()
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        else:
            logger.error(f"Task failed after {self.max_retries} retries: {exc}")
            # self.apply_async(args=[contents, file_name], queue='failed')
            raise

# @celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
# def insert_to_chroma(self, file_id: str, texts: Iterable[str], metadatas: List[dict] = None, ids: List[str] = None):
#     # 写向量数据库任务
#     try:
#         # ids = ids[:10]
#         # metadatas = metadatas[:10]
#         # texts = texts[:10]
#         chroma_client_collection.add(ids, metadatas=metadatas, documents=texts)
#         # TODO 把以下两句放到callback里
#         mongodb_client.update_fileinfo(file_id, {'upload_state_vectorstore': 'done'})
#         mongodb_client.update_upload_state(file_id)
#         return ids
#     except Exception as exc:
#         logger.error(f"Task failed: {exc}, retrying...")
#         traceback.print_exc()
#         if self.request.retries < self.max_retries:
#             raise self.retry(exc=exc)
#         else:
#             logger.error(f"Task failed after {self.max_retries} retries: {exc}")
#             # self.apply_async(args=[file_name, texts, metadatas, ids], queue='failed')
#             raise

# @celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
# def insert_to_chroma(self, file_id: str):
#     # 写向量数据库任务
#     try:
#         chunks = mongodb_client.get_chunks(file_id)
#         documents = []
#         ids = []
#         metadatas = []
#         for chunk in chunks:
#             # logger.info(f"{chunk['id']} {chunk['metadata']} {len(chunk['context'])} {len(chunk['chunk'])}")
#             ids.append(chunk['id'])
#             metadatas.append(chunk['metadata'])
#             documents.append(f"{chunk['context']}\n{chunk['chunk']}")
#         logger.info(f"{len(ids)} {len(documents)} {len(metadatas)}")
#         chroma_client_collection.add(ids, metadatas=metadatas, documents=documents)
#         # TODO 把以下两句放到callback里
#         mongodb_client.update_fileinfo(file_id, {'upload_state_vectorstore': 'done'})
#         mongodb_client.update_upload_state(file_id)
#         return ids
#     except Exception as exc:
#         logger.error(f"Task failed: {exc}, retrying...")
#         traceback.print_exc()
#         if self.request.retries < self.max_retries:
#             raise self.retry(exc=exc)
#         else:
#             logger.error(f"Task failed after {self.max_retries} retries: {exc}")
#             # self.apply_async(args=[file_name, texts, metadatas, ids], queue='failed')
#             raise


@celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
def task_start(self, file_id: str, **kw):
    _task_start(file_id, **kw)

@celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
def task_end(self, file_id: str, **kw):
    _task_end(file_id, **kw)


def _task_end(file_id: str, **kw):
    # 结束完整任务时执行
    if not file_id:
        return
    task_done_date = datetime.now()
    data = mongodb_client.get_one_data('files', {'file_id': file_id})
    if not data:
        return
    task_start_date: datetime = data['task_start_date']
    d = task_done_date - task_start_date
    mongodb_client.update_data('files', {
        'task_done_date': task_done_date,
        'cost_seconds': d.total_seconds(),
    }, {'file_id': file_id})

def _task_start(file_id: str, **kw):
    # 开始完整任务时执行
    if not file_id:
        return
    task_start_date = datetime.now()
    mongodb_client.update_data('files', {
        'task_start_date': task_start_date,
        'cost_seconds': 0,
    }, {'file_id': file_id})

@task_prerun.connect
def task_prerun_notifier(task_id, task, *args, **kwargs):
    logger.info("From task_prerun_notifier ==> Running")
    # logger.info(f'{task_id}, {task}, {kwargs}')
    # logger.info(type(kwargs))
    # logger.info(kwargs.get('kwargs').get('file_id'))
    # logger.info(f'{task_id}, {kwargs.get("file_id")}')
    # instance_attributes = [attr for attr in dir(task) if not callable(getattr(task, attr)) and not attr.startswith("__")]
    # logger.info(str(instance_attributes))
    file_id = kwargs.get('kwargs').get('file_id', '')
    if kwargs.get('kwargs').get('task_start'):
        _task_start(file_id)
    # 统计任务执行时间
    result = mongodb_client.get_one_data('task_info', {'task_id': task_id})
    start_time = datetime.now()
    if result:
        mongodb_client.update_data('task_info', {
            'start_time': start_time,
        }, {'task_id': task_id})
    else:
        mongodb_client.insert_data('task_info', {
            'task_id': task_id,
            'task_name': task.name,
            # 'start_time': datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S'),
            'start_time': start_time,
            'cost_seconds': 0,
            'file_id': file_id
        })
 
 
@task_postrun.connect
def task_postrun_notifier(task_id, task, retval, *args, **kwargs):
    logger.info("From task_postrun_notifier ==> Ok, done!")
    file_id = kwargs.get('kwargs').get('file_id', '')
    if kwargs.get('kwargs').get('task_end'):
       _task_end(file_id)
    # 统计任务执行时间
    end_time = datetime.now()
    data = mongodb_client.get_one_data('task_info', {'task_id': task_id})
    if not data:
        return
    start_time: datetime = data['start_time']
    d = end_time - start_time
    mongodb_client.update_data('task_info', {
        'end_time': end_time,
        'cost_seconds': d.total_seconds(),
    }, {'task_id': task_id})
 
 
@task_success.connect
def task_sucess_notifier(result, *args, **kwargs):
    logger.info("From task_success_notifier ==> Task run successfully!")
    # logger.info(f'{result}, {args}, {kwargs}')
    # TODO 统计任务执行结果
 
 
@task_failure.connect
def task_failure_notifier(task_id, exception, traceback, einfo, *args, **kwargs):
    logger.info("From task_failure_notifier ==> Task failed successfully!")
    # logger.info(f'{task_id}, {exception}, {traceback}, {einfo}, {args}, {kwargs}')
    # TODO 统计任务执行结果
