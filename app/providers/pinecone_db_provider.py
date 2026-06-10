import hashlib
import re
import time
import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import requests
from pinecone import Pinecone, ServerlessSpec
from app.configurations.config import PINECONE_API_KEY, CHUNK_THRESHOLD
from app.models.models import IndexConfig, VideoTranscriptRequest, QueryRequest, UpsertRequest, DataItem
from app.providers.vector_db_provider import VectorDBProvider
from app.services.text_splitter_service import TextSplitterService
from app.services.embedding_service import EmbeddingService
from app.services.file_processor_service import FileProcessorService

TRANSCRIPT_TARGET_SECONDS = 75
TRANSCRIPT_TARGET_CHARS = 900
TRANSCRIPT_MIN_SECONDS_FOR_PUNCTUATION = 35
TRANSCRIPT_MIN_CHARS_FOR_PUNCTUATION = 450
TRANSCRIPT_MAX_SECONDS = 120
TRANSCRIPT_MAX_CHARS = 1400


class PineconeDBProvider(VectorDBProvider):
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.text_splitter = TextSplitterService()
        self.embedding_service = EmbeddingService()
        self.file_processor = FileProcessorService()

    def create_index(self, config: IndexConfig):
        self.pc.create_index(
            name=config.index_name,
            dimension=config.dimension,
            metric=config.metric,
            spec=ServerlessSpec(
                cloud=config.cloud,
                region=config.region
            )
        )

        while not self.pc.describe_index(config.index_name).status['ready']:
            time.sleep(1)

    def upsert_data(self, index_name: str, upsert_request: UpsertRequest):
        index = self.pc.Index(index_name)
        
        all_records = []
        
        for record in upsert_request.records:
            all_records.append(record)
            
            if record.file_urls:
                file_records = self.file_processor.process_file_urls_to_records(
                    record.file_urls, 
                    record.id, 
                    record.metadata
                )
                all_records.extend(file_records)
        
        modified_request = UpsertRequest(
            namespace=upsert_request.namespace,
            records=all_records
        )
        
        if len(all_records) > 100:
            return self._upsert_data_batched(index, modified_request)
        
        return self._upsert_data_optimized(index, modified_request)

    def upsert_video_transcript(
        self,
        index_name: str,
        namespace: str,
        transcript_request: VideoTranscriptRequest,
    ):
        transcript_json = self._download_transcript_json(transcript_request.transcript_json_url)
        chunks = self._build_video_transcript_chunks(
            transcript_json,
            transcript_request,
        )
        if not chunks:
            raise ValueError(f"No transcript chunks found for {transcript_request.id}")

        embeddings = self._create_embeddings_for_chunks(chunks)
        vectors = self._build_vectors_from_chunks_and_embeddings(chunks, embeddings)

        index = self.pc.Index(index_name)
        self._delete_existing_chunks_by_original_id(index, namespace, transcript_request.id)
        self._upsert_vectors_in_batches(index, namespace, vectors)

        return {
            "index_name": index_name,
            "namespace": namespace,
            "id": transcript_request.id,
            "chunks_indexed": len(vectors),
        }

    def _download_transcript_json(self, transcript_json_url: str) -> Dict[str, Any]:
        response = requests.get(transcript_json_url, timeout=30)
        response.raise_for_status()
        return response.json()

    def _build_video_transcript_chunks(
        self,
        transcript_json: Dict[str, Any],
        transcript_request: VideoTranscriptRequest,
    ) -> List[Dict[str, Any]]:
        items = transcript_json.get("results", {}).get("items", [])
        chunks = self._split_transcribe_items_by_timestamp(items)
        timestamp = int(time.time() * 1000)
        total_chunks = len(chunks)
        vector_id_prefix = self._stable_transcript_vector_id_prefix(transcript_request.id)
        playback_url = transcript_request.hls_url or transcript_request.video_url
        base_metadata = {
            **(transcript_request.metadata or {}),
            "data_type": transcript_request.data_type,
            "source_type": transcript_request.source_type,
            "original_id": transcript_request.id,
            "transcript_json_url": transcript_request.transcript_json_url,
            "created_at": timestamp,
        }
        if transcript_request.hls_url:
            base_metadata["hls_url"] = transcript_request.hls_url
        if transcript_request.video_url:
            base_metadata["source_video_url"] = transcript_request.video_url
        if playback_url:
            base_metadata["playback_url"] = playback_url

        results = []
        for index, chunk in enumerate(chunks):
            chunk_id = f"{vector_id_prefix}_chunk_{index}"
            prev_chunk_id = f"{vector_id_prefix}_chunk_{index - 1}" if index > 0 else None
            next_chunk_id = f"{vector_id_prefix}_chunk_{index + 1}" if index < total_chunks - 1 else None
            start_time = chunk["start_time"]
            end_time = chunk["end_time"]

            metadata = {
                **base_metadata,
                "chunk_index": index,
                "total_chunks": total_chunks,
                "chunk_size": len(chunk["text"]),
                "chunk_preview": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"],
                "start_time": start_time,
                "end_time": end_time,
            }
            if playback_url:
                metadata["video_url"] = f"{playback_url}?t={int(start_time)}"
            if prev_chunk_id:
                metadata["prev_chunk_id"] = prev_chunk_id
            if next_chunk_id:
                metadata["next_chunk_id"] = next_chunk_id

            results.append({
                "id": chunk_id,
                "text": chunk["text"],
                "metadata": metadata,
            })

        return results

    def _split_transcribe_items_by_timestamp(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        current_text = ""
        current_start: Optional[float] = None
        current_end: Optional[float] = None
        last_token_was_punctuation = False

        def flush_current():
            nonlocal current_text, current_start, current_end, last_token_was_punctuation
            text = current_text.strip()
            if text and current_start is not None and current_end is not None:
                chunks.append({
                    "text": text,
                    "start_time": current_start,
                    "end_time": current_end,
                })
            current_text = ""
            current_start = None
            current_end = None
            last_token_was_punctuation = False

        for item in items:
            item_type = item.get("type")
            content = self._transcribe_item_content(item)
            if not content:
                continue

            if item_type == "punctuation":
                if current_text:
                    current_text += content
                    last_token_was_punctuation = True
                    if self._should_close_transcript_chunk(
                        current_text,
                        current_start,
                        current_end,
                        ended_with_punctuation=True,
                    ):
                        flush_current()
                continue

            start_time = self._parse_transcribe_time(item.get("start_time"))
            end_time = self._parse_transcribe_time(item.get("end_time"))
            if start_time is None or end_time is None:
                continue

            if current_start is None:
                current_start = start_time

            current_end = end_time
            if current_text and not last_token_was_punctuation:
                current_text += " "
            elif current_text and last_token_was_punctuation:
                current_text += " "
            current_text += content
            last_token_was_punctuation = False

            if self._should_close_transcript_chunk(
                current_text,
                current_start,
                current_end,
                ended_with_punctuation=False,
            ):
                flush_current()

        flush_current()
        return chunks

    def _should_close_transcript_chunk(
        self,
        text: str,
        start_time: Optional[float],
        end_time: Optional[float],
        ended_with_punctuation: bool,
    ) -> bool:
        if start_time is None or end_time is None:
            return False

        duration = end_time - start_time
        text_length = len(text)
        if duration >= TRANSCRIPT_MAX_SECONDS or text_length >= TRANSCRIPT_MAX_CHARS:
            return True
        if ended_with_punctuation and (
            duration >= TRANSCRIPT_TARGET_SECONDS
            or text_length >= TRANSCRIPT_TARGET_CHARS
            or duration >= TRANSCRIPT_MIN_SECONDS_FOR_PUNCTUATION
            or text_length >= TRANSCRIPT_MIN_CHARS_FOR_PUNCTUATION
        ):
            return True
        return False

    def _transcribe_item_content(self, item: Dict[str, Any]) -> str:
        alternatives = item.get("alternatives") or []
        if not alternatives:
            return ""
        return str(alternatives[0].get("content", "")).strip()

    def _parse_transcribe_time(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _stable_transcript_vector_id_prefix(self, original_id: str) -> str:
        digest = hashlib.sha1(original_id.encode("utf-8")).hexdigest()[:16]
        safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", original_id).strip("_")[:80]
        return f"{safe_id}_{digest}_transcript"

    def _delete_existing_chunks_by_original_id(self, index, namespace: str, original_id: str):
        try:
            index.delete(
                filter={"original_id": {"$eq": original_id}},
                namespace=namespace,
            )
        except Exception:
            pass

    def _upsert_vectors_in_batches(self, index, namespace: str, vectors: List[Dict[str, Any]]):
        batch_size = 100
        with ThreadPoolExecutor(max_workers=20) as upsert_executor:
            futures = []
            for i in range(0, len(vectors), batch_size):
                batch_vectors = vectors[i:i + batch_size]
                future = upsert_executor.submit(index.upsert, vectors=batch_vectors, namespace=namespace)
                futures.append(future)

            for future in as_completed(futures):
                future.result()

    def _process_and_upsert_batch(self, index, records, namespace):
        """Procesa un solo lote de registros y los carga a Pinecone."""
        chunks = self._process_records_to_chunks(records)
        if not chunks:
            return

        embeddings = self._create_embeddings_for_chunks(chunks)
        vectors = self._build_vectors_from_chunks_and_embeddings(chunks, embeddings)
        
        # Hacemos el upsert en lotes paralelos para máxima velocidad
        upsert_batch_size = 100
        with ThreadPoolExecutor(max_workers=20) as upsert_executor:
            futures = []
            for i in range(0, len(vectors), upsert_batch_size):
                batch_vectors = vectors[i:i + upsert_batch_size]
                future = upsert_executor.submit(index.upsert, vectors=batch_vectors, namespace=namespace)
                futures.append(future)
            
            # Esperar a que todos los lotes de upsert terminen
            for future in as_completed(futures):
                future.result()
    
    def _upsert_data_optimized(self, index, upsert_request: UpsertRequest):
        def _prepare_vectors():
            chunks = self._process_records_to_chunks(upsert_request.records)
            if not chunks:
                return None
            embeddings = self._create_embeddings_for_chunks(chunks)
            return self._build_vectors_from_chunks_and_embeddings(chunks, embeddings)

        with ThreadPoolExecutor(max_workers=2) as executor:
            delete_future = executor.submit(self._delete_existing_document_chunks, index, upsert_request)
            vectors_future = executor.submit(_prepare_vectors)
            
            delete_future.result()
            vectors = vectors_future.result()
            
            if vectors:
                batch_size = 100
                with ThreadPoolExecutor(max_workers=20) as upsert_executor:
                    futures = []
                    for i in range(0, len(vectors), batch_size):
                        batch_vectors = vectors[i:i + batch_size]
                        future = upsert_executor.submit(index.upsert, vectors=batch_vectors, namespace=upsert_request.namespace)
                        futures.append(future)
                    
                    for future in as_completed(futures):
                        future.result()
    
    def _upsert_data_batched(self, index, upsert_request: UpsertRequest):
        self._delete_existing_document_chunks(index, upsert_request)

        batch_size = 50 
        records = upsert_request.records
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(0, len(records), batch_size):
                batch_records = records[i:i + batch_size]
                future = executor.submit(
                    self._process_and_upsert_batch, 
                    index, 
                    batch_records, 
                    upsert_request.namespace
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error en un lote de upsert: {e}")
    
    def _process_records_to_chunks(self, records):
        all_chunks = []
        timestamp = int(time.time() * 1000)
        
        for record in records:
            # Verificar si record.data["text"] es una lista (para archivos grandes procesados en chunks)
            text_content = record.data.get('text') if hasattr(record, 'data') and isinstance(record.data, dict) else None
            is_chunked_list = isinstance(text_content, list)

            if is_chunked_list: # Este es el caso para CSV
                # `text_content` es ahora una lista de listas de diccionarios
                for chunk_group in text_content:
                    for item in chunk_group:
                        # Cada 'item' tiene 'text' para embedding y 'metadata' completos
                        chunk_id = f"{record.id}_csv_row_{len(all_chunks)}"
                        all_chunks.append({
                            "id": chunk_id,
                            "text": item["text"],
                            "metadata": {
                                **record.metadata, # Metadatos base del record
                                **item["metadata"]  # Metadatos completos de la fila
                            }
                        })
            else: # Caso para TXT, MD, etc.
                file_type = record.metadata.get('file_type', 'unknown')
                
                stream_chunk_index = 0
                # ... (el resto de la lógica original para archivos no-CSV)
                # Aquí asumimos que text_content es una lista de strings
                if isinstance(text_content, list):
                    for content_chunk in text_content:
                        chunk_id = f"{record.id}_stream_{stream_chunk_index}"
                        
                        if len(content_chunk) > CHUNK_THRESHOLD:
                            sub_chunks = self.text_splitter.split_text_with_metadata(
                                text=content_chunk,
                                original_id=chunk_id,
                                metadata=record.metadata
                            )
                            all_chunks.extend(sub_chunks)
                        else:
                            enhanced_metadata = {
                                **record.metadata,
                                "original_id": record.id,
                                "chunk_index": stream_chunk_index,
                                "total_chunks": len(text_content),
                                "chunk_size": len(content_chunk),
                                "created_at": timestamp
                            }
                            all_chunks.append({
                                "id": chunk_id,
                                "text": content_chunk,
                                "metadata": enhanced_metadata
                            })
                        stream_chunk_index += 1
                else:
                    # Procesamiento para datos que no son de archivo (texto plano, etc.)
                    combined_text = self.text_splitter.combine_data_values(record.data)
                    
                    if len(combined_text) > CHUNK_THRESHOLD:
                        chunks = self.text_splitter.split_text_with_metadata(
                            text=combined_text,
                            original_id=record.id,
                            metadata=record.metadata
                        )
                        all_chunks.extend(chunks)
                    else:
                        enhanced_metadata = {
                            **record.metadata,
                            "original_id": record.id,
                            "chunk_index": 0,
                            "total_chunks": 1,
                            "chunk_size": len(combined_text),
                            "created_at": timestamp
                        }
                        all_chunks.append({
                            "id": record.id,
                            "text": combined_text,
                            "metadata": enhanced_metadata
                        })
        
        return all_chunks
    
    def _create_embeddings_for_chunks(self, chunks):
        texts = [chunk["text"] for chunk in chunks]
        return self.embedding_service.create_embeddings(texts)
    
    def _build_vectors_from_chunks_and_embeddings(self, chunks, embeddings):
        return [
            {
                "id": chunk["id"],
                "values": embedding,
                "metadata": {
                    **chunk["metadata"],
                    "text": chunk["text"]
                }
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
    
    def _delete_existing_document_chunks(self, index, upsert_request):
        original_ids = [record.id for record in upsert_request.records]
        
        if not original_ids:
            return
            
        try:
            index.delete(
                filter={
                    "$or": [
                        {"original_id": {"$in": original_ids}},
                        {"original_record_id": {"$in": original_ids}}
                    ]
                },
            namespace=upsert_request.namespace
        )
        except Exception:
            pass

    def search(self, index_name: str, query_request: QueryRequest):
        index = self.pc.Index(index_name)

        results_to_return = []
        if query_request.ids:
            query_results = index.fetch(query_request.ids, query_request.namespace)
            for vector_id, vector_data in query_results['vectors'].items():
                metadata = vector_data.get('metadata', {})
                text_content = metadata.pop('text', '')
                
                results_to_return.append({
                    'id': vector_data['id'],
                    'score': None,
                    'metadata': metadata,
                    'vector': vector_data['values'],
                    'text': text_content
                })
        else:
            query_embedding = self.embedding_service.create_single_embedding(query_request.query)

            query_results = index.query(
                namespace=query_request.namespace,
                vector=query_embedding,
                top_k=query_request.top_k,
                include_values=True,
                include_metadata=True,
                filter=query_request.metadata_filter
            )

            for match in query_results['matches']:
                metadata = match.get('metadata', {})
                text_content = metadata.pop('text', '')
                
                results_to_return.append({
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': metadata,
                    'vector': match['values'],
                    'text': text_content
                })

        return results_to_return
    
    def ensure_namespace_exists(self, index_name: str, namespace: str):
        try:
            index = self.pc.Index(index_name)
            
            index.query(
                namespace=namespace,
                vector=[0.0] * 1536,
                top_k=1,
                include_metadata=False
            )
            
            return {"message": f"Namespace '{namespace}' está listo en índice '{index_name}'", "exists": True}
        except Exception as e:
            if "dimension" in str(e).lower():
                return {"message": f"Namespace '{namespace}' se creará automáticamente en el primer upsert", "exists": False}
            else:
                raise Exception(f"Error con namespace: {str(e)}")
