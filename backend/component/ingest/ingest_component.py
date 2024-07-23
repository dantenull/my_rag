from pathlib import Path
from vectorstores.chroma import Chroma
from llama_index.core.readers.base import BaseReader
from reader.file.pdf_reader import MyPdfReader
from typing import List, Dict, Union
from db.mongodb import MyMongodb
from db.es_client import ElasticsearchClient
from component.llm.llm_component import LLMComponent
from settings.settings import Settings
import time
from constants import PROJECT_ROOT_PATH
from datetime import datetime
# from celery import Celery
from schema import Content
from celery_app import process_async_insert_to_es, insert_to_mongodb, insert_to_chroma


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

# class MyEmbeddings(Embeddings):
#     def __init__(self, embeddings_path) -> None:
#         super().__init__()
#         self.model = embeddings_path
#         self.tiktoken_model_name = embeddings_path

# celery_app = Celery(
#     'tasks', 
#     broker='redis://localhost:6379/0', 
#     backend='redis://localhost:6379/0', 
# )


class IngestComponent:
    def __init__(self, llm_component: LLMComponent, db: MyMongodb, settings: Settings, es_client: ElasticsearchClient) -> None:
        self.settings = settings
        self.llm_component = llm_component.llm
        self.db = db
        self.vectorstore = Chroma(self.llm_component.tokenizer, settings.chroma_collection)
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
    #     file_info = MyPdfReader().load_data_by_semantic(file_path, FakeEmbeddings(model=self.settings.llm_model_path, size=self.settings.llm_size), self.settings.llm_size)
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
    
    def ingest_file_local(self, file_path: Union[str, Path], **kw):
        # def process_metadata(m: Dict) -> None:
        #     m['model'] = self.llm_component.model_name

        if isinstance(file_path, str):
            file = Path(file_path)
        else:
            file = file_path
        extension = file_path.suffix
        if extension != '.pdf':
            print('只能传pdf')
            return 
        file_info = MyPdfReader().load_data_by_page(file_path)
       
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
        info = {
            'name': file.name,
            'stem': file.stem,
            'extension': extension,
            'path': file_path if isinstance(file_path, str) else '',
            'max_content_level': file_info.max_content_level,
            'max_page_num': file_info.max_page_num,
            'upload_date': datetime.now().strftime('%Y.%m.%d %H:%M:%S'),
            'upload_state': 'waiting',
            'upload_state_no_sql': 'waiting' if file_info.contents else 'done',
            'upload_state_elasticsearch': 'waiting',
            'upload_state_vectorstore': 'waiting' if self.vectorstore.embedding_model.model_name in ['openai', 'zhipuai'] else 'done',
            'author': file_info.author,
            'creator': file_info.creator,
            'producer': file_info.producer,
            'subject': file_info.subject,
            'title': file_info.title,
            'modification_date': file_info.modification_date,
            'creation_date': file_info.creation_date,
            **kw
        }
        file_id = self.db.insert_fileinfo(info)
        if file_info.contents:
            contents = []
            for content in file_info.contents:
                content.file_id = file_id
                # if isinstance(content, Content):
                #     content = content.model_dump()
                contents.append(content.model_dump())
            task_mongo = insert_to_mongodb.apply_async(args=[contents, file.name])

        # 存elaticsearch库
        # self.es_client.insert('upload_files', file_info.documents)
        documents = []
        for document in file_info.documents:
            document.file_id = file_id
            documents.append(document.model_dump())
        task_es = process_async_insert_to_es.apply_async(args=[file.name, 'upload_files', documents])

        # 存向量数据库
        documents = [d for d in file_info.documents if d.text]
        ids = [document.doc_id for document in documents]
        metadatas = []
        for document in documents:
            document.metadata['model'] = self.llm_component.model_name
            metadatas.append(document.metadata)
        # map(process_metadata, metadatas)
        texts = [document.text for document in documents]
        # self.vectorstore.add_texts(texts, metadatas, ids)
        task_vectorstore_id = ''
        if self.vectorstore.embedding_model.model_name in ['openai', 'zhipuai']:
            # 只有使用openai等调接口的模型时才进行异步保存
            # 因为本地模型如果异步保存的话，需要加载模型
            # TODO 将本地模型改成服务
            task_vectorstore = insert_to_chroma.apply_async(args=[file.name, texts, metadatas, ids])
            task_vectorstore_id = task_vectorstore.id
        else:
            self.vectorstore.add_texts(texts, metadatas, ids)
        # print(task_es.id, task_mongo.id, task_vectorstore.id)
        return [
            {'task_id': task_es.id, 'name': 'es'},
            {'task_id': task_mongo.id, 'name': 'mongo'},
            {'task_id': task_vectorstore_id, 'name': 'vectorstore'},
        ]
    
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
    
    def get_documents(self, file_name: str, pages_index: list[int]):
        return self.vectorstore.get_documents(file_name, pages_index)
    
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
    
    def get_file_info(self, file_name: str):
        return self.vectorstore.get_file_info(file_name)

    async def delete_by_file(self, file_name: str) -> None:
        where = {
            "file_name": {'$eq': file_name},
        }
        self.vectorstore.delete(where=where)
        self.db.delete_fileinfo(file_name)
        self.db.delete_contents(file_name)
        await self.es_client.async_delete_file('upload_files', file_name)
