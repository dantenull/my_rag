from pydantic import BaseModel, Field
import uuid
from typing import Any, List, Optional, Union, Dict


class Content(BaseModel):
    file_name: str
    title: str
    page_num: int
    level: int
    last_title: str


class Document(BaseModel):
    doc_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
    )
    text: str = Field(
        default="", 
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
    )


class FileInfo(BaseModel):
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

