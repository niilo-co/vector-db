from abc import ABC, abstractmethod
from app.models.models import VideoTranscriptRequest, QueryRequest, UpsertRequest


class VectorDBProvider(ABC):
    @abstractmethod
    def create_index(self, config):
        pass

    @abstractmethod
    def upsert_data(self, index_name: str, upsert_request: UpsertRequest):
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
    def search(self, index_name: str, query_request: QueryRequest):
        pass
    
    @abstractmethod
    def ensure_namespace_exists(self, index_name: str, namespace: str):
        pass
