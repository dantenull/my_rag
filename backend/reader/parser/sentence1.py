from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from constants import DEFAULT_CHUNK_SIZE, DEFAULT_PARAGRAPH_SEP, CHUNKING_REGEX, SENTENCE_CHUNK_OVERLAP

# from llama_index.core.bridge.pydantic import Field, PrivateAttr
# from llama_index.core.callbacks.base import CallbackManager
# from llama_index.core.callbacks.schema import CBEventType, EventPayload
# from llama_index.core.constants import DEFAULT_CHUNK_SIZE
# from llama_index.core.node_parser.interface import (
#     MetadataAwareTextSplitter,
# )
# from llama_index.core.node_parser.node_utils import default_id_func
# from llama_index.core.node_parser.text.utils import (
#     split_by_char,
#     split_by_regex,
#     split_by_sentence_tokenizer,
#     split_by_sep,
# )
# from llama_index.core.utils import get_tokenizer


@dataclass
class _Split:
    text: str  # the split text
    is_sentence: bool  # save whether this is a full sentence
    token_size: int  # token length of split text

# class _Chunk:
#     text: str  
#     # TODO 先按顺序从1往后赋值，之后改为页数。
#     seq: int  # 在全文中的顺序
#     text_with_context: str  


def split_text_keep_separator(text: str, separator: str) -> List[str]:
    """Split text with separator and keep the separator at the end of each split."""
    parts = text.split(separator)
    result = [separator + s if i > 0 else s for i, s in enumerate(parts)]
    return [s for s in result if s]


def split_by_sep(sep: str, keep_sep: bool = True) -> Callable[[str], List[str]]:
    """Split text by separator."""
    if keep_sep:
        return lambda text: split_text_keep_separator(text, sep)
    else:
        return lambda text: text.split(sep)


def split_by_regex(regex: str) -> Callable[[str], List[str]]:
    """Split text by regex."""
    import re

    return lambda text: re.findall(regex, text)


def split_by_char() -> Callable[[str], List[str]]:
    """Split text by character."""
    return lambda text: list(text)


class SentenceSplitter:
    """Parse text with a preference for complete sentences.

    In general, this class tries to keep sentences and paragraphs together. Therefore
    compared to the original TokenTextSplitter, there are less likely to be
    hanging sentences or parts of sentences at the end of the node chunk.
    """

    # chunk_size: int = Field(
    #     default=DEFAULT_CHUNK_SIZE,
    #     description="The token chunk size for each chunk.",
    #     gt=0,
    # )
    # chunk_overlap: int = Field(
    #     default=SENTENCE_CHUNK_OVERLAP,
    #     description="The token overlap of each chunk when splitting.",
    #     ge=0,
    # )
    # separator: str = Field(
    #     default=" ", description="Default separator for splitting into words"
    # )
    # paragraph_separator: str = Field(
    #     default=DEFAULT_PARAGRAPH_SEP, description="Separator between paragraphs."
    # )
    # secondary_chunking_regex: Optional[str] = Field(
    #     default=CHUNKING_REGEX, description="Backup regex for splitting into sentences."
    # )

    def __init__(
        self,
        separator: str = " ",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
        tokenizer: Optional[Callable] = None,
        paragraph_separator: str = DEFAULT_PARAGRAPH_SEP,
        secondary_chunking_regex: Optional[str] = CHUNKING_REGEX,
    ):
        """Initialize with parameters."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._tokenizer = tokenizer

        self._split_fns = [
            split_by_sep(paragraph_separator),
            # self._chunking_tokenizer_fn,
        ]

        if secondary_chunking_regex:
            self._sub_sentence_split_fns = [
                split_by_regex(secondary_chunking_regex),
                split_by_sep(separator),
                split_by_char(),
            ]
        else:
            self._sub_sentence_split_fns = [
                split_by_sep(separator),
                split_by_char(),
            ]

    @classmethod
    def class_name(cls) -> str:
        return "SentenceSplitter"

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, chunk_size=self.chunk_size)

    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """
        _Split incoming text and return chunks with overlap size.

        Has a preference for complete sentences, phrases, and minimal overlap.
        """
        if text == "":
            return [text]

        splits = self._split(text, chunk_size)
        chunks = self._merge(splits, chunk_size)

        return chunks

    def _split(self, text: str, chunk_size: int) -> List[_Split]:
        r"""Break text into splits that are smaller than chunk size.

        The order of splitting is:
        1. split by paragraph separator
        2. split by chunking tokenizer (default is nltk sentence tokenizer)
        3. split by second chunking regex (default is "[^,\.;]+[,\.;]?")
        4. split by default separator (" ")

        """
        token_size = self._token_size(text)
        if token_size <= chunk_size:
            return [_Split(text, is_sentence=True, token_size=token_size)]

        text_splits_by_fns, is_sentence = self._get_splits_by_fns(text)

        text_splits = []
        for text_split_by_fns in text_splits_by_fns:
            token_size = self._token_size(text_split_by_fns)
            if token_size <= chunk_size:
                text_splits.append(
                    _Split(
                        text_split_by_fns,
                        is_sentence=is_sentence,
                        token_size=token_size,
                    )
                )
            else:
                recursive_text_splits = self._split(
                    text_split_by_fns, chunk_size=chunk_size
                )
                text_splits.extend(recursive_text_splits)
        return text_splits

    def _merge(self, splits: List[_Split], chunk_size: int) -> List[str]:
        """Merge splits into chunks."""
        chunks: List[str] = []
        cur_chunk: List[Tuple[str, int]] = []  # list of (text, length)
        last_chunk: List[Tuple[str, int]] = []
        cur_chunk_len = 0
        new_chunk = True

        def close_chunk() -> None:
            nonlocal chunks, cur_chunk, last_chunk, cur_chunk_len, new_chunk

            chunks.append("".join([text for text, length in cur_chunk]))
            last_chunk = cur_chunk
            cur_chunk = []
            cur_chunk_len = 0
            new_chunk = True

            # add overlap to the next chunk using the last one first
            # there is a small issue with this logic. If the chunk directly after
            # the overlap is really big, then we could go over the chunk_size, and
            # in theory the correct thing to do would be to remove some/all of the
            # overlap. However, it would complicate the logic further without
            # much real world benefit, so it's not implemented now.
            if len(last_chunk) > 0:
                last_index = len(last_chunk) - 1
                while (
                    last_index >= 0
                    and cur_chunk_len + last_chunk[last_index][1] <= self.chunk_overlap
                ):
                    text, length = last_chunk[last_index]
                    cur_chunk_len += length
                    cur_chunk.insert(0, (text, length))
                    last_index -= 1

        while len(splits) > 0:
            cur_split = splits[0]
            if cur_split.token_size > chunk_size:
                raise ValueError("Single token exceeded chunk size")
            if cur_chunk_len + cur_split.token_size > chunk_size and not new_chunk:
                # if adding split to current chunk exceeds chunk size: close out chunk
                close_chunk()
            else:
                if (
                    cur_split.is_sentence
                    or cur_chunk_len + cur_split.token_size <= chunk_size
                    or new_chunk  # new chunk, always add at least one split
                ):
                    # add split to chunk
                    cur_chunk_len += cur_split.token_size
                    cur_chunk.append((cur_split.text, cur_split.token_size))
                    splits.pop(0)
                    new_chunk = False
                else:
                    # close out chunk
                    close_chunk()

        # handle the last chunk
        if not new_chunk:
            chunk = "".join([text for text, length in cur_chunk])
            chunks.append(chunk)

        # run postprocessing to remove blank spaces
        return self._postprocess_chunks(chunks)

    def _postprocess_chunks(self, chunks: List[str]) -> List[str]:
        """Post-process chunks.
        Remove whitespace only chunks and remove leading and trailing whitespace.
        """
        new_chunks = []
        for chunk in chunks:
            stripped_chunk = chunk.strip()
            if stripped_chunk == "":
                continue
            new_chunks.append(stripped_chunk)
        return new_chunks

    def _token_size(self, text: str) -> int:
        return len(self._tokenizer(text))

    def _get_splits_by_fns(self, text: str) -> Tuple[List[str], bool]:
        for split_fn in self._split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                return splits, True

        for split_fn in self._sub_sentence_split_fns:
            splits = split_fn(text)
            if len(splits) > 1:
                break

        return splits, False


# if __name__ == '__main__':
#     from PyPDF2 import PdfReader
#     from functools import partial
#     import tiktoken 

#     file_path = 'C:\\Users\\86176\\Documents\\AI\\rag_learn\\经纬华夏.pdf'

#     reader = PdfReader(file_path)
#     # meta = reader.metadata
#     s = ''
#     for page in reader.pages:
#         text = page.extract_text()
#         if not text:
#             continue
#         s += text
    
#     enc = tiktoken.encoding_for_model('gpt-4o-mini')
#     tokenizer = partial(enc.encode, allowed_special="all")

#     splitter = SentenceSplitter(
#         chunk_size=DEFAULT_CHUNK_SIZE,
#         chunk_overlap=SENTENCE_CHUNK_OVERLAP,
#         paragraph_separator=DEFAULT_PARAGRAPH_SEP,
#         secondary_chunking_regex=CHUNKING_REGEX,
#         tokenizer=tokenizer
#     )
#     chunks = splitter.split_text(s)
#     print(chunks)
