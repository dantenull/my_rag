from fastapi import APIRouter, Depends, Request
from service.chat.chat_service import ChatService
from pydantic import BaseModel, Field


chat_router = APIRouter()

class ChatBody(BaseModel):
    message: str

class QueryBody(BaseModel):
    query: str = None
    file_name: str
    n: int = 1
    answer: str = ''

@chat_router.post("/chat")
def chat(request: Request, message: ChatBody):
    service = request.state.injector.get(ChatService)
    resp = service.chat(message.message)
    return resp

@chat_router.post("/similarity_search")
def similarity_search(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = service.similarity_search(query.query, query.file_name, query.n)
    return resp

@chat_router.post("/similarity_search1")
def similarity_search1(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = service.similarity_search1(query.query, query.file_name, query.n)
    return resp

@chat_router.post("/similarity_search_by_es")
def similarity_search(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = service.similarity_search_by_es(query.query, query.file_name, query.n)
    return resp

@chat_router.post("/query_document")
def query_document(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = service.query_document_by_content(query.query, query.file_name)
    return resp

@chat_router.post("/evaluation_test_dateset_by_distance")
def evaluation_test_dateset_by_distance(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = service.evaluation_test_dateset_by_distance(query.file_name, query.n)
    return resp

@chat_router.post("/evaluation_test_dateset_by_openai")
def evaluation_test_dateset_by_openai(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = service.evaluation_test_dateset_by_openai(query.file_name, query.n)
    return resp

@chat_router.post("/evaluation_doc_by_distance")
def evaluation_doc_by_distance(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = service.evaluation_doc_by_distance(query.query, query.answer, query.file_name, query.n)
    return resp

@chat_router.post("/evaluation_doc_by_openai")
def evaluation_doc_by_openai(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = service.evaluation_doc_by_openai(query.query, query.file_name, query.n)
    return resp
