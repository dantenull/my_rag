from PyPDF2 import PdfReader
from pathlib import Path
from schema import Document, Content, FileInfo
from typing import Any, List, Optional, Union, Tuple
from constants import DEFAULT_CHUNK_SIZE, DEFAULT_PARAGRAPH_SEP, CHUNKING_REGEX, SENTENCE_CHUNK_OVERLAP
import re
# from llama_index.core.node_parser import SentenceSplitter
# from tools import title_process
# from langchain_experimental.text_splitter import SemanticChunker
from functools import partial
import tiktoken
from reader.parser.sentence1 import SentenceSplitter
from llms.openai_llm import OpenaiLLM

GET_CONTEXT_PROMPT = """
以下是全文：

{whole_document}

以下是我们想要放置在整个文档中的块:

{chunk_content}

请简要概述此段落在整个文档中的位置，以便改善对该段落的搜索检索。仅提供简要概述，不要提供其他内容。
"""


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

    # def load_data_by_semantic(self, file_path: Union[str, Path], embeddings, chunk_size: int=512) -> FileInfo:
    #     print('--------load_data_by_semantic---------')
    #     if isinstance(file_path, str):
    #         file = Path(file_path)
    #     else:
    #         file = file_path
    #     self.reader = PdfReader(file_path)
    #     self.file_name = file.name
    #     text = ''
    #     for page in self.reader.pages:
    #         text += page.extract_text()
    #     num_chunks = len(text) / chunk_size 
    #     print(f'num_chunks: {num_chunks}')
    #     text_splitter = SemanticChunker(embeddings, number_of_chunks=int(num_chunks))
    #     chunks = text_splitter.create_documents([text])
    #     print(f'chunks: {len(chunks)}')
    #     metadata = {
    #         'file_name': self.file_name,
    #     }
    #     chunks = [Document(text=chunk.page_content, metadata=metadata) for chunk in chunks]
    #     return FileInfo(documents=chunks)

    # def _get_chunk_context(self, chunk: str, whole_document: str) -> str:
    #     openai_llm = OpenaiLLM(
    #         'gpt-4o-mini', 
    #         custom_embedding_model=None,
    #         api_base='https://api.gptsapi.net/v1'
    #     )
    #     result = openai_llm.chat(GET_CONTEXT_PROMPT.format(whole_document=whole_document, chunk_content=chunk))
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

    def load_data(self, file_path: str) -> FileInfo:
        if isinstance(file_path, str):
            file = Path(file_path)
        else:
            file = file_path

        self.reader = PdfReader(file_path)
        self.file_name = file.name
        meta = self.reader.metadata
        s = ''
        for page in self.reader.pages:
            text = page.extract_text()
            if not text:
                continue
            s += text
        
        enc = tiktoken.encoding_for_model('gpt-4o-mini')
        tokenizer = partial(enc.encode, allowed_special="all")

        splitter = SentenceSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=SENTENCE_CHUNK_OVERLAP,
            paragraph_separator=DEFAULT_PARAGRAPH_SEP,
            secondary_chunking_regex=CHUNKING_REGEX,
            tokenizer=tokenizer
        )
        chunks = splitter.split_text(s)

        documents = []
        for i, chunk in enumerate(chunks):
            # start, end = self._get_start_end_index_around(5, len(chunks), i)
            # chunk_context = self._get_chunk_context(chunk, '\n'.join(chunks[start: end]))
            metadata = {
                'file_name': self.file_name,
                'seq': i + 1,
                'words_num': len(chunk),
                # 'origin_chunk': chunk,
                # 'chunk_context': chunk_context,
            }
            # text = f'{chunk_context}\n{chunk}'
            documents.append(Document(text=chunk, metadata=metadata))

        return FileInfo(
            author=meta.author, 
            title=meta.title, 
            documents=documents, 
            max_page_num=len(self.reader.pages)
        )
    
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
        meta = self.reader.metadata
        content_list = self.reader.outline
        content_list1 = []
        max_content_level = self._process_content1(content_list, content_list1, 1)
        contents = self._process_content2(content_list1)
        documents = []
        # for page in self.reader.pages:
        #     # parts = []
        #     text = page.extract_text()
        #     # text = "".join(parts)
        #     metadata = {
        #         'file_name': self.file_name,
        #         'page_num': self.reader.get_page_number(page),
        #         'words_num': len(text),
        #     }
        #     documents.append(Document(text=text, metadata=metadata))
        chunks_text = []
        chunks_page_num = []
        for page in self.reader.pages:
            # 分段落和句子
            # TODO 处理分页时句子断裂的情况
            text = page.extract_text()
            if not text:
                continue
            page_num = self.reader.get_page_number(page)
            paragraph_split = text.split(DEFAULT_PARAGRAPH_SEP)
            for p in paragraph_split:
                chunk_split = re.findall(CHUNKING_REGEX, p)
                for c in chunk_split:
                    c = c.replace(' ', '')
                    if not c:
                        continue
                    chunks_text.append(c)
                    chunks_page_num.append(page_num)
        
        current_chunk_length = 0
        current_index = 0
        for i, chunk in enumerate(chunks_text):
            # 合并句子到指定的大小
            chunk_length = len(chunk)
            current_chunk_length += chunk_length
            if current_chunk_length < DEFAULT_CHUNK_SIZE:
                continue
            else:
                if current_index != i:
                    text = ''.join(chunks_text[current_index: i])
                    page_nums = ','.join(set([str(p) for p in chunks_page_num[current_index: i]]))
                    current_index = i + 1
                    current_chunk_length -= chunk_length
                else:
                    text = chunks_text[i]
                    page_nums = str(chunks_page_num[i])
                    current_index = i
                    current_chunk_length = chunk_length
                metadata = {
                    'file_name': self.file_name,
                    'page_nums': page_nums,
                    'words_num': len(text),
                }
                documents.append(Document(text=text, metadata=metadata))
        
        # modification_date = meta.modification_date.strftime('%Y-%m-%d %H:%M:%S') if meta.modification_date else ''
        # creation_date = meta.creation_date.strftime('%Y-%m-%d %H:%M:%S') if meta.creation_date else ''
        return FileInfo(
            author=meta.author, 
            # creator=meta.creator, 
            # producer=meta.producer, 
            # subject=meta.subject, 
            title=meta.title, 
            # modification_date=modification_date,
            # creation_date=creation_date,
            contents=contents, 
            max_content_level=max_content_level, 
            documents=documents, 
            max_page_num=len(self.reader.pages)
        )

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


if __name__ == '__main__':
    pdf_reader = MyPdfReader()
    file_path = 'C:\\Users\\86176\\Documents\\AI\\rag_learn\\经纬华夏.pdf'
    result = pdf_reader.load_data(file_path)
    print(result)
