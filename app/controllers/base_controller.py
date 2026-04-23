from fastapi import APIRouter, HTTPException, Depends
from app.models.models import IndexConfig, UpsertRequest, QueryRequest
from app.services.vector_db_service_interface import VectorDBServiceInterface

router = APIRouter(
    prefix="/api/ms/vector-db",
    tags=["vector-db"]
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
