from abc import ABC, abstractmethod
from app.models.models import IndexConfig, VideoTranscriptRequest, UpsertRequest, QueryRequest
from typing import List, Dict, Any


class VectorDBServiceInterface(ABC):
    @abstractmethod
    def create_index(self, provider_name: str, config: IndexConfig):
        pass

    @abstractmethod
    def upsert_data(self, provider_name: str, index_name: str, upsert_request: UpsertRequest):
        pass

    @abstractmethod
    def upsert_video_transcript(
        self,
        index_name: str,
        namespace: str,
        transcript_request: VideoTranscriptRequest,
    ):
        pass

    @abstractmethod
    def search(self, provider_name: str, index_name: str, query_request: QueryRequest):
        pass
    
    @abstractmethod
    def ensure_namespace_exists(self, provider_name: str, index_name: str, namespace: str):
        pass
    
    @abstractmethod
    def get_chunk_with_context(self, provider_name: str, index_name: str, chunk_id: str, namespace: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_document_chunks(self, provider_name: str, index_name: str, original_id: str, namespace: str) -> List[Dict[str, Any]]:
        pass
