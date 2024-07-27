from constants import DEFAULT_CHUNK_SIZE, DEFAULT_PARAGRAPH_SEP, CHUNKING_REGEX, SENTENCE_CHUNK_OVERLAP
from typing import List
import re


class SentenceSplitter:
    '''
    将文本按句子分割。

    模仿 llamaindex 的 SentenceSplitter 类。
    '''

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = SENTENCE_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        # 按句子分割
        chunks_text = []
        paragraph_split = text.split(DEFAULT_PARAGRAPH_SEP)
        for p in paragraph_split:
            chunk_split = re.findall(CHUNKING_REGEX, p)
            for c in chunk_split:
                # c = c.replace(' ', '')
                if not c:
                    continue
                chunks_text.append(c)
        
        current_chunk_length = 0
        current_index = 0
        texts = []
        for i, chunk in enumerate(chunks_text):
            # 合并句子到指定的大小
            chunk_length = len(chunk)
            current_chunk_length += chunk_length
            if current_chunk_length < DEFAULT_CHUNK_SIZE:
                continue
            else:
                if current_index != i:
                    t = ''.join(chunks_text[current_index: i])
                    current_index = i + 1
                    current_chunk_length -= chunk_length
                else:
                    t = chunks_text[i]
                    current_index = i
                    current_chunk_length = chunk_length
                texts.append(t)
        return texts
