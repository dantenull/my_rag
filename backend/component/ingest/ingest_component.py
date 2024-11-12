from pathlib import Path
from vectorstores.chroma import Chroma
from llama_index.core.readers.base import BaseReader
from reader.file.pdf_reader import MyPdfReader
from typing import List, Dict, Union, Tuple
from db.mongodb import MyMongodb
from db.es_client import ElasticsearchClient
from component.llm.llm_component import LLMComponent
from component.embeddings.embeddings_component import EmbeddingsComponent
from settings.settings import Settings
import time
from constants import PROJECT_ROOT_PATH
from datetime import datetime
# from celery import Celery
from schema import Content, FileInfo
from celery_app import process_async_insert_to_es, insert_to_mongodb, insert_chunks_to_mongodb, test_task, insert_to_milvus, insert_chunk_to_mongodb
from tools import get_embedding_model
import uuid
from celery import group, chain
from vectorstores.milvus import Milvus


def _try_loading_included_file_formats() -> dict[str, type[BaseReader]]:
    try:
        from llama_index.readers.file.docs import (  # type: ignore
            DocxReader,
            HWPReader,
            PDFReader,
        )
        from llama_index.readers.file.epub import EpubReader  # type: ignore
        from llama_index.readers.file.image import ImageReader  # type: ignore
        from llama_index.readers.file.ipynb import IPYNBReader  # type: ignore
        from llama_index.readers.file.markdown import MarkdownReader  # type: ignore
        from llama_index.readers.file.mbox import MboxReader  # type: ignore
        from llama_index.readers.file.slides import PptxReader  # type: ignore
        from llama_index.readers.file.tabular import PandasCSVReader  # type: ignore
        from llama_index.readers.file.video_audio import (  # type: ignore
            VideoAudioReader,
        )
        # from langchain_community.document_loaders import PyPDFLoader
    except ImportError as e:
        raise ImportError("`llama-index-readers-file` package not found") from e

    default_file_reader_cls: dict[str, type[BaseReader]] = {
        ".hwp": HWPReader,
        ".pdf": PDFReader,
        # ".pdf": MyPdfReader,
        ".docx": DocxReader,
        ".pptx": PptxReader,
        ".ppt": PptxReader,
        ".pptm": PptxReader,
        ".jpg": ImageReader,
        ".png": ImageReader,
        ".jpeg": ImageReader,
        ".mp3": VideoAudioReader,
        ".mp4": VideoAudioReader,
        ".csv": PandasCSVReader,
        ".epub": EpubReader,
        ".md": MarkdownReader,
        ".mbox": MboxReader,
        ".ipynb": IPYNBReader,
    }
    return default_file_reader_cls

FILE_READER_CLS = _try_loading_included_file_formats()


class IngestComponent:
    def __init__(self, llm_component: LLMComponent, embeddings_component: EmbeddingsComponent, db: MyMongodb, settings: Settings, es_client: ElasticsearchClient) -> None:
        self.settings = settings
        self.llm_component = llm_component.llm
        self.embeddings = embeddings_component.embeddings
        self.db = db
        # self.embeddings_model = get_embedding_model(
        #     settings.embeddings.model, settings.embeddings.model_name, self.llm_component.tokenizer)
        self.embeddings_model = settings.embeddings.model_name
        self.vectorstore = Milvus(settings.milvus.uri, settings.milvus.port, settings.milvus.database)
        # self.vectorstore = Chroma(self.llm_component.tokenizer, settings.chroma.collection, self.embeddings_model)
        self.es_client = es_client
    
    # 根据语义分割文件，但不适配中文
    # def ingest_file_by_semantic(self, file_path: str):
    #     def process_metadata(m: Dict) -> None:
    #         m['model'] = self.llm_component.model_name
    #         m['file_name'] = file.name
        
    #     file = Path(file_path)
    #     extension = file.suffix
    #     if extension != '.pdf':
    #         print('只能传pdf')
    #         return 
    #     file_info = MyPdfReader().load_data_by_semantic(file_path, FakeEmbeddings(model=self.settings.llm.model, size=self.settings.llm_size), self.settings.llm_size)
    #     documents = [d for d in file_info.documents if d.text]
    #     ids = [document.doc_id for document in documents]
    #     metadatas = [document.metadata for document in documents]
    #     map(process_metadata, metadatas)
    #     texts = [document.text for document in documents]
    #     self.vectorstore.add_texts(texts, metadatas, ids)
    
    def ingest_file(self,file_name: str, fileb: bytes, **kw):
        # def mkdir(p):
        #     if not os.path.exists(p):
        #         os.mkdir(p)

        t = str(time.time())
        path = PROJECT_ROOT_PATH / 'temp' / str(t)
        path.mkdir(parents=True, exist_ok=True)
        path = path / file_name
        with open(path, 'wb') as f:
            f.write(fileb)
        return self.ingest_file_local(path.absolute(), **kw)
    
    # def _get_chunk_context(self, chunk: str, whole_document: str) -> str:
    #     # openai_llm = OpenaiLLM(
    #     #     'gpt-4o-mini', 
    #     #     custom_embedding_model=None,
    #     #     api_base='https://api.gptsapi.net/v1'
    #     # )
    #     result = self.llm_component.chat(GET_CONTEXT_PROMPT.format(whole_document=whole_document, chunk_content=chunk))
    #     return result
    
    def _get_start_end_index_around(self, n: int, total: int, index: int) -> Tuple[int, Union[int, None]]:
        a = (2 * n + 1)
        if total <= a:
            return 0, None

        if index <= n:
            return 0, a
        
        if index >= (total -n):
            return -a, None

        start = index - 5
        end = start + a
        return start, end

    def ingest_file_local(self, file_path: Union[str, Path], **kw):
        # def process_metadata(m: Dict) -> None:
        #     m['model'] = self.llm_component.model_name
        # task = test_task.apply_async()
        # print(task.id)
        # return

        if isinstance(file_path, str):
            file = Path(file_path)
        else:
            file = file_path
        extension = file.suffix
        if extension != '.pdf':
            print('只能传pdf')
            return 
        file_info = MyPdfReader().load_data(file_path)
        # print(file_info)
        # return
       
        # 存nosql数据库
        # self.db.insert_fileinfo({
        #     'name': file.name,
        #     'stem': file.stem,
        #     'extension': extension,
        #     'path': file_path if isinstance(file_path, str) else '',
        #     'max_content_level': file_info.max_content_level,
        #     'max_page_num': file_info.max_page_num,
        #     'upload_date': datetime.now().strftime('%Y.%m.%d %H:%M:%S'),
        #     **kw
        # })
        # if file_info.contents:
        #     self.db.insert_contents(file_info.contents, file.name)
        # embedding_model = self._get_embedding_model()
        file_id = str(uuid.uuid4())
        info = {
            'file_id': file_id,
            'name': file.name,
            'stem': file.stem,
            'extension': extension,
            'path': file_path if isinstance(file_path, str) else '',
            'max_content_level': file_info.max_content_level,
            'max_page_num': file_info.max_page_num,
            'upload_date': datetime.now(),
            'upload_state': 'waiting',
            'upload_state_no_sql': 'waiting' if file_info.contents else 'done',
            'upload_state_elasticsearch': 'waiting',
            'upload_state_vectorstore': 'waiting',
            # 'upload_state_vectorstore': 'waiting' if self.vectorstore.embedding_model.model_name in ['openai', 'zhipuai'] else 'done',
            # 'author': file_info.author,
            # 'creator': file_info.creator,
            # 'producer': file_info.producer,
            # 'subject': file_info.subject,
            'title': file_info.title,
            # 'modification_date': file_info.modification_date,
            # 'creation_date': file_info.creation_date,
            'embedding_model': self.embeddings_model,
            **kw
        }
        self.db.insert_fileinfo(info)
        task_mongo_id = ''
        # if file_info.contents:
        #     contents = []
        #     for content in file_info.contents:
        #         content.file_id = file_id
        #         # if isinstance(content, Content):
        #         #     content = content.model_dump()
        #         contents.append(content.model_dump())
        #     task_mongo = insert_to_mongodb.apply_async(args=[contents, file.name])
        #     task_mongo_id = task_mongo.id

        # 存elaticsearch库
        # self.es_client.insert('upload_files', file_info.documents)
        documents = []
        for document in file_info.documents:
            document.file_id = file_id
            document.embedding_model = self.embeddings_model
            # print(document.model_dump())
            documents.append(document.model_dump())
        task_es = process_async_insert_to_es.apply_async(kwargs={
            'file_id': file_id, 
            'index_name': 'upload_files', 
            'data': documents,
            'task_start': True,
        })
        # task_es = process_async_insert_to_es.delay(
        #     file_id, 
        #     'upload_files', 
        #     documents)

        # 存向量数据库
        documents = [d for d in file_info.documents if d.text]
        chunks = []
        texts = [document.text for document in documents]
        for i, document in enumerate(documents):
            start, end = self._get_start_end_index_around(5, len(texts), i)
            document.metadata['embeddings_model'] = self.embeddings_model
            chunks.append({
                'chunk': document.text,
                'document': '\n'.join(texts[start: end]),
                'metadata': document.metadata,
                'id': document.doc_id,
                'file_id': file_id,
                'file_name': file.name,
            })
        # task_vectorstore = chain(group([insert_chunk_to_mongodb.apply_async(args=[chunk]) for chunk in chunks]), insert_to_chroma.apply_async(args=[file_id]))
        # task_vectorstore = chain(group([insert_chunk_to_mongodb.apply_async(args=[chunk]) for chunk in chunks[:10]]), insert_to_milvus.apply_async(args=[file_id]))
        # task_vectorstore = chain(
        #     group([insert_chunk_to_mongodb.apply_async(args=[chunk], kwargs={'file_id': file_id}) for chunk in chunks[:2]]), 
        #     insert_to_milvus.si(kwargs={'file_id': file_id, 'task_end': True})()
        # ).apply_async()
        task_vectorstore = chain(
            group([insert_chunk_to_mongodb.si(chunk_info=chunk, file_id=file_id) for chunk in chunks[:2]]), 
            insert_to_milvus.si(file_id = file_id, task_end = True)
        ).apply_async()
        # task_vectorstore = chain(
        #     insert_chunks_to_mongodb.apply_async(kwargs={'chunk_infos': chunks[:10],'file_id': file_id}), 
        #     insert_to_milvus.apply_async(kwargs={'file_id': file_id})
        # )
        # task_vectorstore = chain(
        #     insert_chunks_to_mongodb.si(chunks[:10], file_id=file_id), 
        #     insert_to_milvus.si(file_id=file_id)
        # ).apply_async()
        print(task_vectorstore.id)
        # task_vectorstore = self._insert_chroma(file_info, file_id)
        # ids = [document.doc_id for document in documents]
        # metadatas = []
        # for document in documents:
        #     document.metadata['embedding_model'] = self.embeddings_model
        #     metadatas.append(document.metadata)
        # texts = [document.text for document in documents]
        # for i, chunk in enumerate(texts):
        #     start, end = self._get_start_end_index_around(5, len(texts), i)
        #     # chunk_context = self._get_chunk_context(chunk, '\n'.join(texts[start: end]))
        #     # text = f'{chunk_context}\n{chunk}'
        #     chunks.append({
        #         'chunk': chunk,
        #         'document': '\n'.join(texts[start: end]),
        #         'metadata': metadatas[i],
        #         'id': ids[i],
        #     }
        # task_vectorstore_id = ''
        # if self.vectorstore.embedding_model.model_name in ['openai', 'zhipuai']:
        #     # 只有使用openai等调接口的模型时才进行异步保存
        #     # 因为本地模型如果异步保存的话，需要加载模型
        #     # TODO 将本地模型改成服务
        #     task_vectorstore = insert_to_chroma.apply_async(args=[file.name, texts, metadatas, ids])
        #     task_vectorstore_id = task_vectorstore.id
        # else:
        #     self.vectorstore.add_texts(texts, metadatas, ids)
        # print(task_es.id, task_mongo.id, task_vectorstore.id)
        return [
            {'task_id': task_es.id, 'name': 'es'},
            {'task_id': task_mongo_id, 'name': 'mongo'},
            {'task_id': task_vectorstore.id, 'name': 'vectorstore'},
        ]
    
    def _insert_milvus(self, file_info: FileInfo, file_id: str):
        data = [document for document in file_info.documents]
    
    def _insert_chroma(self, file_info: FileInfo, file_id: str):
        documents = [d for d in file_info.documents if d.text]
        chunks = []
        texts = [document.text for document in documents]
        for i, document in enumerate(documents):
            start, end = self._get_start_end_index_around(5, len(texts), i)
            document.metadata['embedding_model'] = self.embeddings_model
            chunks.append({
                'chunk': document.text,
                'document': '\n'.join(texts[start: end]),
                'metadata': document.metadata,
                'id': document.doc_id,
                'file_id': file_id,
            })
        task_vectorstore = chain(group([insert_chunk_to_mongodb.apply_async(args=[chunk]) for chunk in chunks]), insert_to_chroma.apply_async(args=[file_id]))
        return task_vectorstore
    
    # def ingest_file(self, file_path: str):
    #     file = Path(file_path)
    #     extension = file.suffix
    #     reader_cls = FILE_READER_CLS.get(extension)
    #     documents = reader_cls().load_data(file_path)
    #     documents = [d for d in documents if d.text]
        
    #     ids = [document.doc_id for document in documents]
    #     metadatas = [document.metadata for document in documents]
    #     texts = [document.text for document in documents]
    #     return self.vectorstore.add_texts(texts, metadatas, ids)
    
    # def get_documents(self, file_name: str, pages_index: list[int]):
    #     return self.vectorstore.get_documents(file_name, pages_index)
    
    # def file_list(self) -> List[Dict]:
    #     all_data = self.vectorstore.get_all_documents()
    #     metadatas = all_data['metadatas']
    #     file_names = []
    #     for metadata in metadatas:
    #         file_names.append(metadata)
    #     return file_names
    
    def file_list(self) -> List[Dict]:
        all_data = self.db.get_all_files()
        return all_data
    
    # def get_file_info(self, file_name: str):
    #     return self.vectorstore.get_file_info(file_name)

    async def delete_by_file(self, file_id: str) -> None:
        # where = {
        #     "file_id": {'$eq': file_id},
        # }
        # self.vectorstore.delete(where=where)
        self.vectorstore.delete_data(self.settings.embeddings.dim , filter=f'file_id == \'{file_id}\'')
        self.db.delete_fileinfo(file_id)
        self.db.delete_contents(file_id)
        self.db.delete_chunks(file_id)
        await self.es_client.async_delete_file('upload_files', file_id)
