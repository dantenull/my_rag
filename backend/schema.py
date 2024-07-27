from pydantic import BaseModel, Field
import uuid
from typing import Any, List, Optional, Union, Dict
from datetime import datetime


class Content(BaseModel):
    file_id: str = Field(default='')
    file_name: str
    title: str
    page_num: int
    level: int
    last_title: str


class Document(BaseModel):
    file_id: str = Field(default='')
    doc_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
    )
    text: str = Field(default='')
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
    )
    embedding_model: str = Field(default='')


class FileInfo(BaseModel):
    file_id: Union[str, None] = Field(default='')
    author: Union[str, None] = Field(default='')
    creator: Union[str, None] = Field(default='')
    producer: Union[str, None] = Field(default='')
    subject: Union[str, None] = Field(default='')
    title: Union[str, None] = Field(default='')
    modification_date: Union[str, None] = Field(default='')
    creation_date: Union[str, None] = Field(default='')
    contents: Optional[List[Content]] = Field(
        default=None, 
        description='目录'
    )
    max_content_level: Optional[int] = Field(
        default=None, 
        description='最大目录层级'
    )
    max_page_num: Optional[int] = Field(
        default=None, 
        description='总页数'
    )
    documents: List[Document] = Field(
        default=None, 
        description=''
    )
    embedding_model: str = Field(default='')

