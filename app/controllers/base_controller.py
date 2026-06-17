import logging

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from app.configurations.config import (
    VIDEO_TRANSCRIPT_INDEX,
    VIDEO_TRANSCRIPT_NAMESPACE,
    VIDEO_TRANSCRIPT_PROVIDER,
)
from app.models.models import (
    IndexConfig,
    VideoTranscriptAcceptedResponse,
    VideoTranscriptRequest,
    UpsertRequest,
    QueryRequest,
)
from app.services.vector_db_service import VectorDBService
from app.services.vector_db_service_interface import VectorDBServiceInterface

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/ms/vector-db",
    tags=["vector-db"]
)


def get_video_transcript_vector_db_service() -> VectorDBServiceInterface:
    return VectorDBService(VIDEO_TRANSCRIPT_PROVIDER)


def run_video_transcript_indexing(
    vector_db_service: VectorDBServiceInterface,
    transcript_request: VideoTranscriptRequest,
    namespace: str,
) -> None:
    try:
        vector_db_service.upsert_video_transcript(
            VIDEO_TRANSCRIPT_INDEX,
            namespace,
            transcript_request,
        )
    except Exception:
        logger.exception(
            "Video transcript indexing failed for id=%s",
            transcript_request.id,
        )


@router.post("/create_index/{provider_name}")
def create_index(provider_name: str, config: IndexConfig,
                 vector_db_service: VectorDBServiceInterface = Depends()):
    try:
        vector_db_service.create_index(provider_name, config)
        return {"message": f"Índice {config.index_name} creado exitosamente en {provider_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error al crear el índice: " + str(e))


@router.post("/upsert_data/{provider_name}/{index_name}")
def upsert_data(provider_name: str, index_name: str, upsert_request: UpsertRequest,
                vector_db_service: VectorDBServiceInterface = Depends()):
    try:
        vector_db_service.upsert_data(provider_name, index_name, upsert_request)
        return {"message": f"Datos insertados exitosamente en el índice {index_name} de {provider_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error al insertar datos: " + str(e))


@router.post(
    "/upsert_video_transcript",
    response_model=VideoTranscriptAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def upsert_video_transcript(
    transcript_request: VideoTranscriptRequest,
    background_tasks: BackgroundTasks,
    vector_db_service: VectorDBServiceInterface = Depends(get_video_transcript_vector_db_service),
):
    namespace = transcript_request.namespace or VIDEO_TRANSCRIPT_NAMESPACE
    background_tasks.add_task(
        run_video_transcript_indexing,
        vector_db_service,
        transcript_request,
        namespace,
    )
    return VideoTranscriptAcceptedResponse(
        id=transcript_request.id,
        index_name=VIDEO_TRANSCRIPT_INDEX,
        namespace=namespace,
    )


@router.post("/search/{provider_name}/{index_name}")
def search(provider_name: str, index_name: str, query_request: QueryRequest,
           vector_db_service: VectorDBServiceInterface = Depends()):
    try:
        results = vector_db_service.search(provider_name, index_name, query_request)
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error en la búsqueda: " + str(e))


@router.post("/ensure_namespace/{provider_name}/{index_name}/{namespace}")
def ensure_namespace(provider_name: str, index_name: str, namespace: str,
                     vector_db_service: VectorDBServiceInterface = Depends()):
    try:
        result = vector_db_service.ensure_namespace_exists(provider_name, index_name, namespace)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health")
async def health_check():
    return {"status": "healthy"}
