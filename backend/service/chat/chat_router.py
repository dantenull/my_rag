from fastapi import APIRouter, Depends, Request
from service.chat.chat_service import ChatService
from pydantic import BaseModel, Field


chat_router = APIRouter()

class ChatBody(BaseModel):
    message: str

class QueryBody(BaseModel):
    query: str = None
    file_id: str
    n: int = 1
    answer: str = ''

class Eval2wikimultihop(BaseModel):
    file_path: str
    eval_num: int

@chat_router.post("/chat")
def chat(request: Request, message: ChatBody):
    service = request.state.injector.get(ChatService)
    resp = service.chat(message.message)
    return resp

# @chat_router.post("/similarity_search")
# def similarity_search(request: Request, query: QueryBody):
#     service = request.state.injector.get(ChatService)
#     resp = service.similarity_search(query.query, query.file_id, query.n)
#     return resp

# @chat_router.post("/similarity_search1")
# def similarity_search1(request: Request, query: QueryBody):
#     service = request.state.injector.get(ChatService)
#     resp = service.similarity_search1(query.query, query.file_id, query.n)
#     return resp

@chat_router.post("/similarity_search_by_es")
async def similarity_search_by_es(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = await service.similarity_search_by_es(query.query, query.file_id, query.n)
    return resp

@chat_router.post("/mix_search")
async def mix_search(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = await service.mix_search(query.query, query.file_id, query.n)
    return resp

# @chat_router.post("/query_document")
# def query_document(request: Request, query: QueryBody):
#     service = request.state.injector.get(ChatService)
#     resp = service.query_document_by_content(query.query, query.file_id)
#     return resp

@chat_router.post("/evaluation_test_dateset_by_distance")
def evaluation_test_dateset_by_distance(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = service.evaluation_test_dateset_by_distance(query.file_id, query.n)
    return resp

@chat_router.post("/evaluation_test_dateset_by_openai")
def evaluation_test_dateset_by_openai(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = service.evaluation_test_dateset_by_openai(query.file_id, query.n)
    return resp

@chat_router.post("/evaluation_doc_by_distance")
def evaluation_doc_by_distance(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = service.evaluation_doc_by_distance(query.query, query.answer, query.file_id, query.n)
    return resp

@chat_router.post("/evaluation_doc_by_openai")
def evaluation_doc_by_openai(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    resp = service.evaluation_doc_by_openai(query.query, query.file_id, query.n)
    return resp

@chat_router.post("/eval_by_quality")
def eval_by_quality(request: Request, query: QueryBody):
    service = request.state.injector.get(ChatService)
    service.eval_by_quality(query.n)

@chat_router.post("/eval_by_2wikimultihop_process_data")
def eval_by_2wikimultihop_process_data(request: Request, data: Eval2wikimultihop):
    service = request.state.injector.get(ChatService)
    service.eval_by_2wikimultihop_process_data(data.file_path, data.eval_num)

@chat_router.post("/eval_by_2wikimultihop")
async def eval_by_2wikimultihop(request: Request, data: Eval2wikimultihop):
    service = request.state.injector.get(ChatService)
    result = await service.eval_by_2wikimultihop(data.file_path, data.eval_num)
    return result

@chat_router.post("/save_2wikimultihop_data")
def save_2wikimultihop_data(request: Request, data: Eval2wikimultihop):
    service = request.state.injector.get(ChatService)
    service.save_2wikimultihop_data(data.file_path, data.eval_num)
