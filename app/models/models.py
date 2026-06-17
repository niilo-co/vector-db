from typing import List, Optional
from pydantic import BaseModel, Field


class IndexConfig(BaseModel):
    index_name: str
    dimension: int
    metric: str
    cloud: str = "aws"
    region: str = "us-east-1"


class DataItem(BaseModel):
    id: str
    data: dict
    metadata: dict = {}
    file_urls: Optional[List[str]] = []


class UpsertRequest(BaseModel):
    namespace: str
    records: List[DataItem]


class QueryRequest(BaseModel):
    query: str = ""
    ids: Optional[List[str]] = None
    top_k: int = 3
    namespace: str
    metadata_filter: dict = {}


class VideoTranscriptRequest(BaseModel):
    id: str
    transcript_json_url: str
    hls_url: Optional[str] = None
    video_url: Optional[str] = None
    namespace: Optional[str] = None
    data_type: str = "video_transcript"
    source_type: str = "video"
    metadata: dict = Field(default_factory=dict)


class VideoTranscriptAcceptedResponse(BaseModel):
    status: str = "ACCEPTED"
    message: str = "Transcript indexing queued"
    id: str
    index_name: str
    namespace: str
