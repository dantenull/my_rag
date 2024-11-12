import chromadb
from chromadb import Documents, EmbeddingFunction
from typing import Any, Iterable, List, Dict
# from injector import singleton
from embeddings import Embeddings


class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embeddings) -> None:
        super().__init__()
        self.embeddings = embeddings

    def __call__(self, input: Documents) -> List[List[float]]:
        embeddings = []
        if self.embeddings.model_name == 'huggingface':
            self.embeddings.add_special_tokens({'pad_token': '[PAD]'})
            # 这里使用 adding='max_length' 才可以补全，padding=True 不行
            embeddings = [self.embeddings.encode(text, max_length=2048, truncation=True, padding='max_length', verbose=True) for text in input if text]
        elif self.embeddings.model_name in ['openai', 'zhipuai']:
            # embeddings = [self.embeddings.encode(text) for text in input if text]
            embeddings = self.embeddings.encode(input)
        return [list(map(float, e)) for e in embeddings]
 
 
class Chroma:
    _DEFAULT_COLLECTION_NAME = "test"

    def __init__(
        self, 
        embedding_model: Embeddings,
        collection_name: str = _DEFAULT_COLLECTION_NAME,
        embedding_model_name: str = '',
    ) -> None:
        self.embedding_model = embedding_model
        self._client = chromadb.PersistentClient(path='.\\chroma_db_test' + '_' + 'gpt-4o-mini')
        self._collection = self._client.create_collection(
            name=collection_name, embedding_function=MyEmbeddingFunction(embedding_model), get_or_create=True)
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: List[Dict] = None,
        ids: List[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        # TODO 考虑更新文档的情况
        self._collection.add(ids, metadatas=metadatas, documents=texts)
        return ids
    
    def delete(self, **kw) -> None:
        self._collection.delete(**kw)

    # def add_texts(self, file_path: Path):
    #     file_name = file_path.name
    #     file = Path(file_name)
    #     extension = file.suffix
    #     reader_cls = FILE_READER_CLS.get(extension)
    #     documents = reader_cls().load_data(file)
    #     documents = [d for d in documents if d.text]
        
    #     documents_ids = [document.doc_id for document in documents]
    #     documents_metadata = [document.metadata for document in documents]
    #     documents_content = [document.text for document in documents]
    #     self._collection.add(documents_ids, metadatas=documents_metadata, documents=documents_content)
    
    # def add_texts(self, file_path: Path):
    #     print(file_path)
    #     # file_name = file_path.name
    #     # file = Path(file_name)
    #     extension = file_path.suffix
    #     if extension != '.pdf':
    #         print('只能传pdf')
    #         return 
    #     print(file_path.name)
    #     print(file_path.root)
    #     reader_cls = FILE_READER_CLS.get(extension)
    #     documents = reader_cls(file_path.name).load_and_split()
    #     print('--------------------------')
    #     print(documents[1:3])
    #     documents = [d for d in documents if d.page_content]
        
    #     documents_ids = [str(doc.metadata["page"]) for doc in documents]
    #     documents_metadata = [doc.metadata for doc in documents]
    #     documents_content = [doc.page_content for doc in documents]
    #     self._collection.add(documents_ids, metadatas=documents_metadata, documents=documents_content)
    
    def query_collection(self, query: str, file_name: str = '', n_results: int = 1, **kwargs):
        if not file_name:
            return self._collection.query(
                query_texts=[query],
                n_results=n_results,
                **kwargs,
            )
        else:
            return self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"file_name": {'$eq': file_name}},
                **kwargs,
            )

    def get_all_documents(self):
        return self._collection.peek(self._collection.count())
    
    def get_file_info(self, file_name: str):
        return self._collection.get(
            where={"file_name": {'$eq': file_name}},
        )
    
    def get(self, where):
        return self._collection.get(
            where=where,
        )
    
    def get_documents(self, file_name: str, pages_index: list[int]):
        # pages_index = [str(p) for p in pages_index]
        all_documents = self.get_all_documents()
        documents = all_documents['documents']
        metadatas = all_documents['metadatas']
        docs = []
        docs_index = []
        for i, metadata in enumerate(metadatas):
            # print(metadata)
            if file_name == metadata.get('file_name') and metadata.get('page_num') in pages_index:
                docs_index.append(i)
        for i, document in enumerate(documents):
            if i in docs_index:
                docs.append({'document': document, 'metadata': metadatas[i]})
        return docs
