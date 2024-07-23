from pymongo import MongoClient
from settings.settings import Settings
from injector import inject, singleton
from typing import List, Dict, Optional
from schema import Content
from tools import content_title_process
import uuid


class MyMongodbBase:
    def __init__(self, port: int, db_name: str) -> None:
        self._client = MongoClient("localhost", port)
        self.db = self._client[db_name]
        self.file_collection = self.db.get_collection('files')
        self.content_collection = self.db.get_collection('contents')
    
    def get_collection(self, collection_name: str):
        return self.db[collection_name]
    
    def insert_fileinfo(self, fileinfo: Dict) -> str:
        file_id = str(uuid.uuid4())
        file = self.file_collection.find_one({'file_id': file_id})
        if file:
            return self.insert_fileinfo(fileinfo)
        else:
            fileinfo['file_id'] = file_id
            self.file_collection.insert_one(fileinfo)
            return file_id
        # if self.get_fileinfo(file_name) and file_name:
        #     self.file_collection.update_one({'name': file_name}, fileinfo)
        # else:
        #     self.file_collection.insert_one(fileinfo)
        # self.delete_fileinfo(file_name)
        # self.file_collection.insert_one(fileinfo)
    
    def insert_contents(self, contents: List[Content], file_name: str):
        self.delete_contents(file_name)
        # print(contents)
        contents = [c.model_dump() if isinstance(c, Content) else c for c in contents]
        self.content_collection.insert_many(contents)
    
    def delete_fileinfo(self, file_name: str) -> None: 
        self.file_collection.delete_one({'name': file_name})
    
    def delete_contents(self, file_name: str) -> None: 
        self.content_collection.delete_many({'file_name': file_name})
    
    def get_fileinfo(self, file_name: str) -> Dict:
        fileinfo = self.file_collection.find_one({'name': file_name})
        return fileinfo
    
    def get_file_max_content_level(self, file_name: str) -> int:
        fileinfo = self.file_collection.find_one({'name': file_name})
        return fileinfo.get('max_content_level')

    def get_contents(self, file_name: str, level: Optional[int], last_titles: Optional[List]) -> List[str]:
        fileinfo = self.get_fileinfo(file_name)
        max_content_level = fileinfo.get('max_content_level')
        if level > max_content_level:
            return
        result = []
        if not last_titles:
            last_titles = ['']
        for last_title in last_titles:
            filter = {
                'file_name': file_name, 
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
    
    def get_all_files(self) -> List[Dict]:
        files = self.file_collection.find()
        result = []
        for file in files:
            result.append({
                'upload_date': file.get('upload_date'),
                'name': file.get('name'),
                'max_page_num': file.get('max_page_num'),
                'size': file.get('size'),
                'upload_state': file.get('upload_state'),
            })
        return result
    
    def update_fileinfo(self, file_name: str, update: Dict):
        result = self.file_collection.update_one({'name': file_name}, {'$set': update})
        # print(result.matched_count, result.modified_count)
        return result.matched_count == result.modified_count

    def update_upload_state(self, file_name:str):
        fileinfo = self.get_fileinfo(file_name)
        upload_state_no_sql = fileinfo.get('upload_state_no_sql')
        upload_state_elasticsearch = fileinfo.get('upload_state_elasticsearch')
        upload_state_vectorstore = fileinfo.get('upload_state_vectorstore')
        # print(fileinfo)
        if (upload_state_no_sql == 'done') and (upload_state_elasticsearch == 'done') and (upload_state_vectorstore == 'done'):
            result = self.update_fileinfo(file_name, {'upload_state': 'done'})
            return result
        return True
            # self.file_collection.update_one({'name': file_name}, {'$set': {'upload_state': 'done'}})


@singleton
class MyMongodb(MyMongodbBase):
    @inject
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.mongodb_port, settings.mongodb_db_name)
    

