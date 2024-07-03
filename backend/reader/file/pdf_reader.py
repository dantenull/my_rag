from PyPDF2 import PdfReader
from pathlib import Path
from schema import Document, Content, FileInfo
from typing import Any, List, Optional, Union, Tuple
# from tools import title_process
from langchain_experimental.text_splitter import SemanticChunker


class MyPdfReader:
    # def __init__(self, file_path: str) -> None:
    #     path = Path(file_path)
    #     self.path = file_path
    #     self.stem = path.stem
    #     self.suffix = path.suffix
    #     self.reader = PdfReader(file_path)
    #     self._init_content()
    
    # def get_content(self) -> Content:
    #     content_list = self.reader.outline
    #     content_list1 = []
    #     max_content_level = self._process_content1(content_list, content_list1, 1)
    #     content = self._process_content2(content_list1)
    #     return Content(contents=content, max_content_level=max_content_level)

    def load_data_by_semantic(self, file_path: Union[str, Path], embeddings, chunk_size: int=512) -> FileInfo:
        print('--------load_data_by_semantic---------')
        if isinstance(file_path, str):
            file = Path(file_path)
        else:
            file = file_path
        self.reader = PdfReader(file_path)
        self.file_name = file.name
        text = ''
        for page in self.reader.pages:
            text += page.extract_text()
        num_chunks = len(text) / chunk_size 
        print(f'num_chunks: {num_chunks}')
        text_splitter = SemanticChunker(embeddings, number_of_chunks=int(num_chunks))
        chunks = text_splitter.create_documents([text])
        print(f'chunks: {len(chunks)}')
        metadata = {
            'file_name': self.file_name,
        }
        chunks = [Document(text=chunk.page_content, metadata=metadata) for chunk in chunks]
        return FileInfo(documents=chunks)
    
    def load_data_by_page(self, file_path: Union[str, Path]) -> FileInfo:
        # def visitor_body(text, cm, tm, fontDict, fontSize):
        #     y = tm[5]
        #     if y <= 0:
        #         parts.append(text)
        
        if isinstance(file_path, str):
            file = Path(file_path)
        else:
            file = file_path
        self.reader = PdfReader(file_path)
        self.file_name = file.name
        content_list = self.reader.outline
        content_list1 = []
        max_content_level = self._process_content1(content_list, content_list1, 1)
        contents = self._process_content2(content_list1)
        documents = []
        for page in self.reader.pages:
            # parts = []
            text = page.extract_text()
            # text = "".join(parts)
            metadata = {
                'file_name': self.file_name,
                'page_num': self.reader.get_page_number(page),
                'words_num': len(text),
            }
            documents.append(Document(text=text, metadata=metadata))
        return FileInfo(contents=contents, max_content_level=max_content_level, documents=documents, max_page_num=len(self.reader.pages))

    def _process_content1(self, content_list :list, content_list1: list, level: int) -> int:
        '''
        处理目录数据第一步。将原来递归表示的目录结果变为一层的列表。
        递归将每一级目录取出，加上层级信息，按顺序放入列表。
        返回最大目录层级。
        '''
        max_num = 1
        for content in content_list:
            if not isinstance(content, list):
                page_number = self.reader.get_destination_page_number(content)
                if len(content.title) > 1:
                    content_list1.append((content.title, page_number, level))
            else:
                max_num = self._process_content1(content, content_list1, level + 1)
                max_num = max_num + 1
        return max_num
    
    def _process_content2(self, content_list: list) -> List[Content]:
        '''
        处理目录数据第二步。
        根据层级信息添加上级目录名称。
        这种结构便于保存，并可根据层级和上级目录名称带出一个目录下所有子目录。
        '''
        last_title = []
        last_level = 0
        result = []
        for content in content_list:
            title, page_num, level = content
            if level == last_level:
                if level != 1:
                    lt = last_title[level - 2]
                else:
                    last_title = [title]
                    last_level = 1
                    lt = ''
            elif level > last_level:
                last_title.append(title)
                lt = last_title[level - 2] if level - 2 >= 0 else ''
            else:
                if level != 1:
                    last_title = last_title[:level] + [title]
                    lt = last_title[level - 2] if level - 2 >= 0 else ''
                else:
                    last_title = [title]
                    last_level = 1
                    lt = ''
            res = Content(
                file_name = self.file_name,
                title = title,
                page_num = page_num,
                level = level,
                last_title = lt,
            )
            result.append(res)
            last_level = level
        return result
