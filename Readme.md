# Vector Database API with FastAPI

This project provides a FastAPI-based API to manage vector database operations, including creating indexes, inserting (upserting) data, and performing similarity or ID-based searches. Currently, it supports **Pinecone** as the vector database provider.

## Features

- Create vector indexes in Pinecone.
- Upsert (insert or update) data into the vector database.
- Search for similar vectors or specific IDs with metadata filtering.

## Requirements

- Python 3.12
- FastAPI == 0.109.1
- Pinecone Client == 3.0.0
- Pydantic == 2.5.2
- Uvicorn == 0.24.0
- python-dotenv
- Docker (opcional)

## Instalación y Ejecución

### Usando Docker

1. Construir la imagen:
```bash
docker build -t vector-db-api .
```

2. Ejecutar el contenedor:
```bash
docker run -d -p 9000:9000 --name vector-db-container vector-db-api
```

3. Verificar que está funcionando:
- La API estará disponible en: http://localhost:9000/docs

4. Comandos útiles de Docker:
```bash
# Ver logs del contenedor
docker logs vector-db-container

# Detener el contenedor
docker stop vector-db-container

# Eliminar el contenedor
docker rm vector-db-container
```

### Instalación Local

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Environment Variables

The following environment variables are required:

- `PINECONE_API_KEY`: Your Pinecone API key.
- `OPENAI_API_KEY`: Your OpenAI API key.

### Advanced Text Processing Configuration

- `CHUNK_SIZE`: Size of text chunks for splitting (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `CHUNK_THRESHOLD`: Minimum text length to trigger splitting (default: 1000)
- `OPENAI_EMBEDDING_MODEL`: OpenAI embedding model to use (default: text-embedding-3-small)

## Features

### Text Splitting
The API automatically splits long texts using a recursive character splitter to:
- Maintain semantic coherence
- Optimize embedding quality
- Handle large documents efficiently
- Preserve metadata across chunks

### OpenAI Embeddings
Uses **text-embedding-3-small** for optimal performance:
- **Cost-effective**: ~10x cheaper than alternatives
- **High quality**: Superior semantic search results
- **1536 dimensions**: Perfect balance of performance and accuracy
- **8,191 token limit**: Handles large texts efficiently

Texts longer than 1000 characters are automatically split into manageable chunks with proper metadata tracking.

### Intelligent Upsert Strategy
The API implements a "replace" strategy instead of simple append:
1. **Clean existing data**: Before inserting new chunks, all existing chunks for the same document ID are automatically deleted
2. **Prevent orphaned chunks**: No leftover chunks from previous versions of the same document
3. **Unique chunk IDs**: Each chunk gets a unique ID with timestamp to prevent collisions
4. **Metadata consistency**: All vectors (chunked or not) include `original_id`, `chunk_index`, `total_chunks`, and `created_at` metadata

**Example workflow**:
- Document "doc1" first upload → `doc1_chunk_0_1234567890`, `doc1_chunk_1_1234567890`
- Document "doc1" update → Automatically deletes previous chunks, creates new ones
- No manual cleanup needed

### Smart Namespace Management
Dedicated namespace service for optimal performance:

#### Dedicated Namespace Endpoints
- `POST /api/namespace/ensure_namespace` - Create/verify namespace before operations
- `GET /api/namespace/namespace_stats/{index}/{namespace}` - Get namespace statistics
- `POST /api/namespace/validate_namespace` - Validate namespace format

#### Performance Optimizations
1. **Separated responsibilities**: Namespace operations are independent from upsert
2. **Optional validation**: Validate only when needed, not on every operation
3. **Faster upserts**: Removed heavy validations from critical path
4. **Lazy creation**: Namespaces created automatically during first upsert

**Usage pattern**:
```bash
# Optional: Pre-validate namespace (recommended for production)
POST /api/namespace/ensure_namespace
{
  "index_name": "my-index",
  "namespace": "production"
}

# Then proceed with fast upserts
POST /api/ms/vector-db/upsert_data/pinecone/my-index
```

**Validation rules**:
- ✅ Valid: `"production"`, `"test-env"`, `"user_123"`
- ❌ Invalid: `"test space"`, `"special@chars"`

## Usage

### 1. Start the Server

Run the FastAPI server using Uvicorn:

```bash
uvicorn main:app --reload
```

### 2. API Endpoints

#### Postman Collection
Para probar los endpoints más fácilmente, puedes usar nuestra colección de Postman:
[Vector DB API Collection](https://sumer-07062021.postman.co/workspace/SUMER~148b538f-9145-4526-8806-bb1cc611d3bd/collection/16642082-d356100e-f0fc-461e-b19f-65935b064b38?action=share&creator=16642082&active-environment=16640760-5fde9fd8-7328-4098-8d82-c9d5fa254624)

#### Create Index

Creates a vector index in Pinecone.

```bash
curl --location 'http://localhost:8000/vector-db/create_index/pinecone' \
--header 'Content-Type: application/json' \
--data '{
    "index_name": "startup",
    "dimension": 1536,
    "metric": "cosine",
    "cloud": "aws",
    "region": "us-east-1"
}'
```

**Note**: Use `1536` dimensions for compatibility with OpenAI's text-embedding-3-small model.

#### Upsert Data

Upserts (inserts or updates) data into the specified namespace of a Pinecone index.

```bash
curl --location 'http://localhost:8000/vector-db/upsert_data/pinecone/startup' \
--header 'Content-Type: application/json' \
--data '{
    "namespace": "products",
    "records": [
        {
            "id": "pc1",
            "data": {"name": "computadora mac", "price": 213131.03131},
            "metadata": {"tags": ["mac"], "category": "tech", "price": 2000,  "id": "pc1"}
        },
        {
            "id": "pc2",
            "data": {"name": "computadora linux", "price": 200.03131},
            "metadata": {"tags": ["linux"], "category": "tech", "price": 1000,  "id": "pc2"}
        }
    ]
}'
```

#### Upsert Video Transcript

Queues an Amazon Transcribe JSON transcript for async indexing into Pinecone.
This endpoint works for LiveKit classes and other MP4 transcripts.
It always uses:

```text
provider = pinecone
index = dynamic-load
namespace = niilo_db
```

```bash
curl --location 'http://localhost:9000/api/ms/vector-db/upsert_video_transcript' \
--header 'Content-Type: application/json' \
--data '{
    "id": "class-live/9junio/9junio_20260610_003733_be1ad19a_retry",
    "transcript_json_url": "https://assets.niilo.co/class-live/9junio/9junio_20260610_003733_be1ad19a_retry/transcripts/transcript.json",
    "hls_url": "https://assets.niilo.co/class-live/9junio/9junio_20260610_003733_be1ad19a_retry/hls/playlist.m3u8",
    "data_type": "live_class_transcript",
    "source_type": "live_class",
    "metadata": {
        "base_prefix": "class-live/9junio/9junio_20260610_003733_be1ad19a_retry",
        "source_mp4_s3": "s3://niilo-prod/class-live/9junio/9junio_20260610_003733_be1ad19a_retry/source/recording.mp4",
        "transcript_json_s3": "s3://niilo-prod/class-live/9junio/9junio_20260610_003733_be1ad19a_retry/transcripts/transcript.json",
        "transcript_text_s3": "s3://niilo-prod/class-live/9junio/9junio_20260610_003733_be1ad19a_retry/transcripts/transcript.txt"
    }
}'
```

The response is immediate:

```json
{
  "status": "ACCEPTED",
  "message": "Transcript indexing queued",
  "id": "class-live/9junio/9junio_20260610_003733_be1ad19a_retry",
  "index_name": "dynamic-load",
  "namespace": "niilo_db"
}
```

The background task downloads the transcript JSON, chunks it with
`start_time` / `end_time`, deletes existing chunks by `original_id`, and
upserts fresh vectors with playback metadata. For non-LiveKit videos, send
`video_url` instead of `hls_url` and use a different `data_type` /
`source_type`, for example `video_transcript` / `video`.

#### Search

Performs a similarity search with optional metadata filters.

```bash
curl --location 'http://localhost:8000/vector-db/search/pinecone/startup' \
--header 'Content-Type: application/json' \
--data '{
    "query": "quiero comprar un pc",
    "top_k": 2,
    "ids": ["pc1", "pc2"],
    "namespace": "products",
    "metadata_filter": {
        "category": {
            "$eq": "tech"
        },
        "price": {
            "$gt": 200
        },
        "id": {
            "$eq": "pc1"
        }
    }
}'
```

## License

This project is licensed under the MIT License.
