'''
celery -A celery_app worker --loglevel=info -P eventlet -Q default,failed
celery -A celery_app flower --address=127.0.0.1 --port=5566
'''

from celery import Celery
from typing import List, Dict, Iterable
from db.es_client import ElasticsearchClientBase
from db.mongodb import MyMongodbBase
from settings.settings import unsafe_typed_settings as rag_settings
import os
import asyncio
from celery.utils.log import get_task_logger
# from celery.schedules import crontab
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from openai import OpenAI
from zhipuai import ZhipuAI
from sentence_transformers import SentenceTransformer

class CustomEmbeddings:
    def __init__(self) -> None:
        # self.model_name = 'custom_embeddings'
        self.custom_embedding_model = rag_settings.custom_embedding_model
        self.model = SentenceTransformer(self.custom_embedding_model, device='cuda')

    def encode(self, inputs: str | List[str], **kw):
        embeddings = self.model.encode(inputs, normalize_embeddings=True)
        return embeddings


custom_embeddings = CustomEmbeddings()


class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self) -> None:
        self.llm = None
        self.llm_mode = rag_settings.llm_mode
        if self.llm_mode == 'openai':
            openai_api_key = os.environ.get('OPENAI_API_KEY')
            self.llm = OpenAI(api_key=openai_api_key)
            self.embeddings_model = 'text-embedding-3-small'
        elif self.llm_mode == 'zhipuai':
            zhipuai_api_key = os.environ.get('ZHIPUAI_API_KEY')
            self.llm = ZhipuAI(api_key=zhipuai_api_key)
            self.embeddings_model = 'embedding-2'
        if self.llm:
            self.embeddings = self.llm.embeddings
        super().__init__()

    def __call__(self, input: Documents) -> List[List[float]]:
        if not self.llm:
            return []
        embeddings = []
        input = [text.replace("\n", " ") for text in input if text]
        # print([len(text) for text in input if len(text) > 512])
        embeddings = self.encode(input)
        return [list(map(float, e)) for e in embeddings]
    
    def encode(self, inputs: List[str]):
        if rag_settings.using_custom_embedding_model:
            return custom_embeddings.encode(inputs)
        else:
            return self._encode(inputs)
    
    # def encode_common(self, inputs: str | List[str], **kw):
    #     model = SentenceTransformer(rag_settings.custom_embedding_model)
    #     embeddings = model.encode(inputs, normalize_embeddings=True)
    #     return embeddings
    
    def _encode(self, inputs: List[str]):
        if self.llm_mode == 'openai':
            return self._openai_encode(inputs)
        elif self.llm_mode == 'zhipuai':
            return self._zhipuai_encode(inputs)
    
    def _openai_encode(self, inputs: List[str]):
        data = self.embeddings.create(input=inputs, model=self.embeddings_model).data
        return [d.embedding for d in data]
    
    def _zhipuai_encode(self, inputs: List[str]):
        return [self.embeddings.create(input=input, model=self.embeddings_model).data[0].embedding for input in inputs]
    
    # def _zhipuai_encode_batch(self, input: List[str]):
    #     input_file_id = 'file_123'
    #     output_file_id = None
    #     self.llm.batches.create(
    #         input_file_id="file_123",
    #         endpoint="/v4/embeddings",
    #         completion_window="24h",
    #     )
    #     completed = False
    #     while not completed:
    #         retrieve = self.llm.batches.retrieve(input_file_id)
    #         if retrieve['status'] != 'completed':
    #             continue
    #         output_file_id = retrieve['output_file_id']
    #         content = self.llm.files.content(output_file_id) 


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
es_client = ElasticsearchClientBase(rag_settings.es_host, rag_settings.es_user, os.getenv('ELASTIC_PASSWORD'))
mongodb_client = MyMongodbBase(rag_settings.mongodb_port, rag_settings.mongodb_db_name)
chroma_client = chromadb.PersistentClient(path='.\\chroma_db_test' + '_' + rag_settings.llm_mode)
chroma_client_collection = chroma_client.get_or_create_collection(rag_settings.chroma_collection, embedding_function=MyEmbeddingFunction())

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

async def async_insert_to_es(file_name: str, index_name: str, data: List[Dict]):
    result = await es_client.async_insert(index_name, data)
    return result

def get_or_create_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

@celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
def process_async_insert_to_es(self, file_name: str, index_name: str, data: List[Dict]):
    # 写elasticsearch任务
    try:
        loop = get_or_create_event_loop()
        async_result = loop.run_until_complete(async_insert_to_es(file_name, index_name, data))
        # TODO 把以下两句放到callback里
        mongodb_client.update_fileinfo(file_name, {'upload_state_elasticsearch': 'done'})
        mongodb_client.update_upload_state(file_name)
        return async_result
    except Exception as exc:
        logger.error(f"Task failed: {exc}, retrying...")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, queue='failed')
        else:
            logger.error(f"Task failed after {self.max_retries} retries: {exc}")
            # move_task_to_failure_queue(self.name, self.request.args, self.request.kwargs, exc)
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
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        else:
            logger.error(f"Task failed after {self.max_retries} retries: {exc}")
            # move_task_to_failure_queue(self.name, self.request.args, self.request.kwargs, exc)
            # self.apply_async(args=[contents, file_name], queue='failed')
            raise

@celery_app.task(bind=True, default_retry_delay=DEFAULT_RETRY_DELAY)
def insert_to_chroma(self, file_name: str, texts: Iterable[str], metadatas: List[dict] = None, ids: List[str] = None):
    # 写向量数据库任务
    try:
        chroma_client_collection.add(ids, metadatas=metadatas, documents=texts)
        # TODO 把以下两句放到callback里
        mongodb_client.update_fileinfo(file_name, {'upload_state_vectorstore': 'done'})
        mongodb_client.update_upload_state(file_name)
        return ids
    except Exception as exc:
        logger.error(f"Task failed: {exc}, retrying...")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
        else:
            logger.error(f"Task failed after {self.max_retries} retries: {exc}")
            # move_task_to_failure_queue(self.name, self.request.args, self.request.kwargs, exc)
            # self.apply_async(args=[file_name, texts, metadatas, ids], queue='failed')
            raise

# @celery_app.task
# def retry_failed_tasks():
#     failed_tasks = celery_app.control.inspect().reserved(queue='failed')
#     for task in failed_tasks:
#         task.apply_async()

# @celery_app.task
# def retry_failed_tasks():
#     logger.info('retry_failed_tasks')
#     # failed_queue = Queue('failed', Exchange('failed'), routing_key='failed')
#     with Connection('redis://localhost:6379/0') as conn:
#         consumer = conn.Consumer(failed_queue, accept=['json'])
#         # producer = conn.Producer()

#         def process_message(body, message):
#             task_name = body['task']
#             task_args = body['args']
#             task_kwargs = body['kwargs']
#             logger.info(f"Retrying task {task_name} with args {task_args} and kwargs {task_kwargs}")
#             task = celery_app.signature(task_name, args=task_args, kwargs=task_kwargs)
#             task.apply_async()
#             message.ack()

#         consumer.register_callback(process_message)
#         consumer.consume()

#         while True:
#             conn.drain_events()

# def move_task_to_failure_queue(task_name, args, kwargs, exc):
#     # failed_queue = Queue('failed', Exchange('failed'), routing_key='failed')
#     with Connection('redis://localhost:6379/0') as conn:
#         producer = conn.Producer()
#         task_info = {
#             'task_name': task_name,
#             'args': args,
#             'kwargs': kwargs,
#             'reason': str(exc),
#         }
#         producer.publish(
#             task_info,
#             exchange=failed_queue.exchange,
#             routing_key=failed_queue.routing_key,
#             declare=[failed_queue]
#         )
#     print(f"Moved task {task_name} to failure queue")
    
