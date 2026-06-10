from app.factories.vector_db_provider_factory import VectorDBProviderFactory
from app.models.models import IndexConfig, VideoTranscriptRequest, UpsertRequest, QueryRequest
from app.services.vector_db_service_interface import VectorDBServiceInterface
from typing import List, Dict, Any, Optional


class VectorDBService(VectorDBServiceInterface):
    def __init__(self, provider_name: str):
        self.provider = VectorDBProviderFactory.get_provider(provider_name)

    def create_index(self, provider_name: str, config: IndexConfig):
        self.provider.create_index(config)

    def upsert_data(self, provider_name: str, index_name: str, upsert_request: UpsertRequest):
        self.provider.upsert_data(index_name, upsert_request)

    def upsert_video_transcript(
        self,
        index_name: str,
        namespace: str,
        transcript_request: VideoTranscriptRequest,
    ):
        return self.provider.upsert_video_transcript(
            index_name,
            namespace,
            transcript_request,
        )

    def search(self, provider_name: str, index_name: str, query_request: QueryRequest):
        return self.provider.search(index_name, query_request)
    
    def ensure_namespace_exists(self, provider_name: str, index_name: str, namespace: str):
        return self.provider.ensure_namespace_exists(index_name, namespace)
    
    def get_chunk_with_context(self, provider_name: str, index_name: str, chunk_id: str, namespace: str) -> Dict[str, Any]:
        query_request = QueryRequest(
            ids=[chunk_id],
            top_k=1,
            namespace=namespace
        )
        
        result = self.provider.search(index_name, query_request)
        if not result or 'matches' not in result or not result['matches']:
            return None
            
        chunk = result['matches'][0]
        chunk_metadata = chunk.get('metadata', {})
        
        context_chunks = {}
        
        # Obtener chunk anterior si existe
        prev_chunk_id = chunk_metadata.get('prev_chunk_id')
        if prev_chunk_id:
            prev_query = QueryRequest(ids=[prev_chunk_id], top_k=1, namespace=namespace)
            prev_result = self.provider.search(index_name, prev_query)
            if prev_result and prev_result.get('matches'):
                context_chunks['previous'] = prev_result['matches'][0]
        
        # Obtener chunk siguiente si existe
        next_chunk_id = chunk_metadata.get('next_chunk_id')
        if next_chunk_id:
            next_query = QueryRequest(ids=[next_chunk_id], top_k=1, namespace=namespace)
            next_result = self.provider.search(index_name, next_query)
            if next_result and next_result.get('matches'):
                context_chunks['next'] = next_result['matches'][0]
        
        return {
            'current_chunk': chunk,
            'context_chunks': context_chunks,
            'full_text': self._combine_chunks_text(context_chunks.get('previous'), chunk, context_chunks.get('next'))
        }
    
    def get_document_chunks(self, provider_name: str, index_name: str, original_id: str, namespace: str) -> List[Dict[str, Any]]:
        query_request = QueryRequest(
            query="",
            top_k=100,
            namespace=namespace,
            metadata_filter={"original_id": original_id}
        )
        
        result = self.provider.search(index_name, query_request)
        if not result or 'matches' not in result:
            return []
            
        chunks = sorted(result['matches'], key=lambda x: x.get('metadata', {}).get('chunk_index', 0))
        return chunks
    
    def _combine_chunks_text(self, prev_chunk: Optional[Dict], current_chunk: Dict, next_chunk: Optional[Dict]) -> str:
        texts = []
        
        if prev_chunk:
            prev_text = prev_chunk.get('metadata', {}).get('chunk_preview', '')
            if prev_text:
                texts.append(f"[Chunk anterior]: {prev_text}")
        
        current_text = current_chunk.get('metadata', {}).get('text', '')
        if current_text:
            texts.append(f"[Chunk actual]: {current_text}")
        
        if next_chunk:
            next_text = next_chunk.get('metadata', {}).get('chunk_preview', '')
            if next_text:
                texts.append(f"[Chunk siguiente]: {next_text}")
        
        return "\n\n".join(texts)
