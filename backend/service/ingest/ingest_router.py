from fastapi import APIRouter, Request, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from service.ingest.ingest_service import IngestService
from celery_app import get_celery_task_status as get_status
import asyncio


ingest_router = APIRouter()

class UploadFileLocal(BaseModel):
    upload_file: str

class FileInfo(BaseModel):
    file_id: str

# class UploadFileEntity(BaseModel):
#     fileb: UploadFile = File()
#     notes: str = Form()

class GetDocumentsBody(BaseModel):
    file_name: str
    pages_index: list[int]

class CeleryTask(BaseModel):
    task_id: str

@ingest_router.post("/upload")
async def ingest_file(request: Request, file: UploadFile):
    service = request.state.injector.get(IngestService)
    return service.ingest_file(file.filename, file.file.read(), size=file.size)

@ingest_router.post("/upload_local")
def upload_local(request: Request, upload_file: UploadFileLocal):
    service = request.state.injector.get(IngestService)
    return service.ingest_file_local(upload_file.upload_file)

# @ingest_router.post("/upload1")
# def ingest_file_by_semantic(request: Request, upload_file: UploadFileLocal):
#     service = request.state.injector.get(IngestService)
#     service.ingest_file_by_semantic(upload_file.upload_file)

@ingest_router.get("/file_list")
def ingest_list(request: Request):
    service = request.state.injector.get(IngestService)
    return service.list_ingested()

# @ingest_router.post("/get_documents")
# def get_documents(request: Request, get_documents_body: GetDocumentsBody):
#     service = request.state.injector.get(IngestService)
#     return service.get_documents(get_documents_body.file_name, get_documents_body.pages_index)

# @ingest_router.post("/get_file_info")
# def get_file_info(request: Request, upload_file: UploadFile):
#     service = request.state.injector.get(IngestService)
#     return service.get_file_info(upload_file.upload_file)

@ingest_router.post("/delete_by_file")
async def delete_by_file(request: Request, file_info: FileInfo):
    service = request.state.injector.get(IngestService)
    await service.delete_by_file(file_info.file_id)

@ingest_router.post("/get_celery_task_status")
def get_celery_task_status(request: Request, celery_task: CeleryTask):
    # service = request.state.injector.get(IngestService)
    return get_status(celery_task.task_id)

@ingest_router.websocket("/ws/get_celery_task_status")
async def get_celery_task_status_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        task_id = data.get('task_id')
        if not task_id:
            return
        last_result = None
        while True:
            task_result = get_status(task_id)
            if (not last_result) or (last_result and (last_result.get('status') != task_result.get('status'))):
                await websocket.send_json(task_result)
                last_result = task_result.copy()
            if task_result.get('status') == 'Completed':
                await websocket.close()
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
