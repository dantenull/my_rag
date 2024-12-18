from pymongo import MongoClient, UpdateOne
from settings.settings import Settings
from injector import inject, singleton
from typing import List, Dict, Optional, Union
from schema import Content
from tools import content_title_process
import uuid
import time
import os


class MyMongodbBase:
    def __init__(self, port: int, db_name: str, username: str) -> None:
        if username:
            password = os.environ.get('MYRAG_MONGODB_PASSWORD')
            self._client = MongoClient("localhost", port, username=username, password=password)
        else:
            self._client = MongoClient("localhost", port)
        self.db = self._client[db_name]
        self.file_collection = self.db.get_collection('files')  # 文件表
        self.content_collection = self.db.get_collection('contents')  # 目录表
        self.chunk_conllection = self.db.get_collection('chunks')  # 文本块表
    
    def get_collection(self, collection_name: str):
        return self.db[collection_name]
    
    def insert_data(self, collection_name: str, data: Union[Dict, List]):
        collection = self.db[collection_name]
        if isinstance(data, list):
            collection.insert_many(data)
        elif isinstance(data, dict):
            collection.insert_one(data)

    def insert_data_by_field(self, collection_name: str, data: List[Dict], field: str):
        # s = time.time()
        operations = []
        for doc in data:
            operations.append(
                UpdateOne(
                    {field: doc[field]},  # 按照字段判断是否存在
                    {"$setOnInsert": doc},  # 如果不存在，插入该文档
                    upsert=True,  # 启用upsert功能
                )
            )
        # 执行批量操作
        if operations:
            collection = self.db[collection_name]
            result = collection.bulk_write(operations)
            # e = time.time()
            # print(f'insert_data_by_field {e - s}')
            return result
    
    def update_data(self, collection_name: str, update, filter):
        collection = self.db[collection_name]
        result = collection.update_one(filter, {'$set': update})
    
    def get_one_data(self, collection_name: str, filter):
        collection = self.db[collection_name]
        data = collection.find_one(filter)
        # print(data)
        return data
    
    def get_data(self, collection_name: str, filter):
        collection = self.db[collection_name]
        datas = collection.find(filter)
        result = []
        for data in datas:
            result.append(data)
        return result
    
    def insert_fileinfo(self, fileinfo: Dict) -> str:
        # file_id = str(uuid.uuid4())
        # file = self.file_collection.find_one({'file_id': file_id})
        # if file:
        #     return self.insert_fileinfo(fileinfo)
        # else:
        #     fileinfo['file_id'] = file_id
        self.file_collection.insert_one(fileinfo)
        # return file_id

        # if self.get_fileinfo(file_name) and file_name:
        #     self.file_collection.update_one({'name': file_name}, fileinfo)
        # else:
        #     self.file_collection.insert_one(fileinfo)
        # self.delete_fileinfo(file_name)
        # self.file_collection.insert_one(fileinfo)
    
    def insert_contents(self, contents: List[Content], file_id: str):
        self.delete_contents(file_id)
        # print(contents)
        contents = [c.model_dump() if isinstance(c, Content) else c for c in contents]
        self.content_collection.insert_many(contents)
    
    def insert_chunk(self, chunk_info: Dict):
        self.chunk_conllection.insert_one(chunk_info)

    def delete_fileinfo(self, file_id: str) -> None: 
        self.file_collection.delete_one({'file_id': file_id})
    
    def delete_contents(self, file_id: str) -> None: 
        self.content_collection.delete_many({'file_id': file_id})
    
    def delete_chunks(self, file_id: str) -> None:
        self.chunk_conllection.delete_many({'file_id': file_id})
    
    def delete_infos(self, collection_name: str, file_id: str):
        collection = self.db[collection_name]
        collection.delete_many({'file_id': file_id})

    def get_fileinfo(self, file_id: str) -> Dict:
        fileinfo = self.file_collection.find_one({'file_id': file_id})
        return fileinfo
    
    def get_file_max_content_level(self, file_id: str) -> int:
        fileinfo = self.file_collection.find_one({'file_id': file_id})
        return fileinfo.get('max_content_level')

    def get_contents(self, file_id: str, level: Optional[int], last_titles: Optional[List]) -> List[str]:
        fileinfo = self.get_fileinfo(file_id)
        max_content_level = fileinfo.get('max_content_level')
        if level > max_content_level:
            return
        result = []
        if not last_titles:
            last_titles = ['']
        for last_title in last_titles:
            filter = {
                'file_id': file_id, 
                'level': level, 
            }
            # if last_title:
            #     filter['last_title'] = last_title
            contents = self.content_collection.find(filter)
            for content in contents:
                if not last_title:
                    result.append(content['title'])
                else:
                    if last_title == content_title_process(content['last_title']):
                        result.append(content['title']) 
        return result
    
    def get_all_files(self, condition: Dict=None) -> List[Dict]:
        files = self.file_collection.find(condition)
        result = []
        for file in files:
            result.append({
                'file_id': file.get('file_id'),
                'upload_date': file.get('upload_date'),
                'name': file.get('name'),
                'max_page_num': file.get('max_page_num'),
                'size': file.get('size'),
                'upload_state': file.get('upload_state'),
            })
        return result
    
    def get_chunks(self, file_id: str) -> List[Dict]:
        result = []
        chunks = self.chunk_conllection.find({'file_id': file_id})
        for chunk in chunks:
            result.append({
                'chunk': chunk['chunk'],
                'document': chunk['document'],
                'context': chunk['context'],
                'id': chunk['id'],
                'metadata': chunk['metadata'],
            })
        return result

    def update_fileinfo(self, file_id: str, update: Dict):
        result = self.file_collection.update_one({'file_id': file_id}, {'$set': update})
        # print(result.matched_count, result.modified_count)
        return result.matched_count == result.modified_count
    
    def get_file_upload_state(self, file_id: str) -> bool:
        fileinfo = self.get_fileinfo(file_id)
        upload_state_no_sql = fileinfo.get('upload_state_no_sql')
        upload_state_elasticsearch = fileinfo.get('upload_state_elasticsearch')
        upload_state_vectorstore = fileinfo.get('upload_state_vectorstore')
        return ((upload_state_no_sql == 'done') and (upload_state_elasticsearch == 'done') and (upload_state_vectorstore == 'done'))

    def update_upload_state(self, file_id:str):
        # fileinfo = self.get_fileinfo(file_id)
        # upload_state_no_sql = fileinfo.get('upload_state_no_sql')
        # upload_state_elasticsearch = fileinfo.get('upload_state_elasticsearch')
        # upload_state_vectorstore = fileinfo.get('upload_state_vectorstore')
        upload_success = self.get_file_upload_state(file_id)
        # print(fileinfo)
        if upload_success:
            result = self.update_fileinfo(file_id, {'upload_state': 'done'})
            return result
        return False
            # self.file_collection.update_one({'name': file_name}, {'$set': {'upload_state': 'done'}})


@singleton
class MyMongodb(MyMongodbBase):
    @inject
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.mongodb.port, settings.mongodb.db_name, settings.mongodb.username)
    

