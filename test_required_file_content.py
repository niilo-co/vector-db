from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from app.controllers.base_controller import upsert_data
from app.models.models import UpsertRequest
from app.providers.pinecone_db_provider import PineconeDBProvider
from app.services.file_processor_service import FileProcessorService
from app.services.text_splitter_service import TextSplitterService
from app.services.vector_db_service import VectorDBService


def request_for(*urls, strict=True):
    return UpsertRequest(
        namespace="creator-docs",
        require_file_content=strict,
        records=[{
            "id": f"document-{i}",
            "data": {"text": "Uploaded placeholder"},
            "metadata": {"owner_id": "owner-1"},
            "file_urls": [url],
        } for i, url in enumerate(urls)],
    )


@pytest.fixture
def provider():
    provider = PineconeDBProvider.__new__(PineconeDBProvider)
    provider.pc = Mock()
    provider.pc.Index.return_value.list.return_value = []
    provider.text_splitter = TextSplitterService()
    provider.file_processor = FileProcessorService()
    provider.file_processor._download_and_process_file = Mock(
        return_value=("Actual document text", {"file_type": ".pdf"}),
    )
    provider.embedding_service = Mock()
    provider.embedding_service.create_embeddings.side_effect = lambda texts: [[1.0, 2.0] for _ in texts]
    return provider


def endpoint(provider, request):
    service = VectorDBService.__new__(VectorDBService)
    service.provider = provider
    return upsert_data("pinecone", "dynamic-agents", request, service)


def test_strict_http_response_confirms_real_text_and_stable_ids(provider):
    request = request_for("https://example.com/one.pdf", "https://example.com/two.pdf")
    response = endpoint(provider, request)
    assert response["indexed_file_urls"] == [record.file_urls[0] for record in request.records]
    index = provider.pc.Index.return_value
    vectors = index.upsert.call_args.kwargs["vectors"]
    assert len(vectors) == 2
    assert all(vector["metadata"]["text"] == "Actual document text" for vector in vectors)
    assert all(vector["metadata"]["owner_id"] == "owner-1" for vector in vectors)
    assert all(vector["values"] for vector in vectors)
    endpoint(provider, request)
    assert [vector["id"] for vector in index.upsert.call_args.kwargs["vectors"]] == [
        vector["id"] for vector in vectors
    ]


@pytest.mark.parametrize("content", ["", " \n\t", [], ["  ", "\n"], [[]]])
def test_empty_extraction_preserves_existing_content(provider, content):
    provider.file_processor._download_and_process_file.return_value = (content, {"file_type": ".pdf"})
    with pytest.raises(HTTPException):
        endpoint(provider, request_for("https://example.com/empty.pdf"))
    provider.pc.Index.return_value.delete.assert_not_called()
    provider.pc.Index.return_value.upsert.assert_not_called()


def test_one_failed_file_prevents_all_writes(provider):
    provider.file_processor._download_and_process_file.side_effect = [
        ("Valid file", {"file_type": ".txt"}),
        RuntimeError("Download failed"),
    ]
    with pytest.raises(HTTPException, match="Download failed"):
        endpoint(provider, request_for("https://example.com/one.txt", "https://example.com/two.txt"))
    provider.pc.Index.return_value.delete.assert_not_called()
    provider.pc.Index.return_value.upsert.assert_not_called()


@pytest.mark.parametrize("result", [[], [[]], RuntimeError("Embedding failed")])
def test_failed_embeddings_preserve_existing_content(provider, result):
    provider.embedding_service.create_embeddings.side_effect = None
    if isinstance(result, Exception):
        provider.embedding_service.create_embeddings.side_effect = result
    else:
        provider.embedding_service.create_embeddings.return_value = result
    with pytest.raises(HTTPException):
        endpoint(provider, request_for("https://example.com/file.pdf"))
    provider.pc.Index.return_value.delete.assert_not_called()
    provider.pc.Index.return_value.upsert.assert_not_called()


@pytest.mark.parametrize("operation", ["list", "delete", "upsert"])
def test_storage_failure_is_not_acknowledged(provider, operation):
    provider.pc.Index.return_value.list.return_value = [["stale_chunk"]]
    getattr(provider.pc.Index.return_value, operation).side_effect = RuntimeError("Storage failed")
    with pytest.raises(HTTPException, match="Storage failed"):
        endpoint(provider, request_for("https://example.com/file.pdf"))
    if operation in ("list", "upsert"):
        provider.pc.Index.return_value.delete.assert_not_called()


@pytest.mark.parametrize("count", [0, 21])
def test_strict_request_is_bounded(provider, count):
    with pytest.raises(HTTPException):
        endpoint(provider, request_for(*(f"https://example.com/{i}.txt" for i in range(count))))
    provider.file_processor._download_and_process_file.assert_not_called()


def test_strict_rejects_multiple_urls_per_record(provider):
    request = request_for("https://example.com/one.txt")
    request.records[0].file_urls.append("https://example.com/two.txt")
    with pytest.raises(HTTPException):
        endpoint(provider, request)
    provider.file_processor._download_and_process_file.assert_not_called()


def test_standard_clients_keep_existing_response_and_processing(provider):
    provider._upsert_data_optimized = Mock()
    response = endpoint(provider, request_for("https://example.com/file.pdf", strict=False))
    assert "indexed_file_urls" not in response
    records = provider._upsert_data_optimized.call_args.args[1].records
    assert len(records) == 2
    assert records[0].data["text"] == "Uploaded placeholder"


def test_csv_rows_use_stable_ids_independent_of_other_documents(provider):
    provider.file_processor._download_and_process_file.return_value = (
        [[{"text": "Actual CSV row", "metadata": {"column": "value"}}]],
        {"file_type": ".csv"},
    )
    request = request_for("https://example.com/one.csv", "https://example.com/two.csv")
    endpoint(provider, request)
    index = provider.pc.Index.return_value
    second_id = index.upsert.call_args.kwargs["vectors"][1]["id"]
    request.records = request.records[1:]
    endpoint(provider, request)
    assert index.upsert.call_args.kwargs["vectors"][0]["id"] == second_id


def test_cleanup_only_removes_stale_ids_after_successful_upsert(provider):
    request = request_for("https://example.com/file.pdf")
    endpoint(provider, request)
    index = provider.pc.Index.return_value
    retained_id = index.upsert.call_args.kwargs["vectors"][0]["id"]
    stale_id = retained_id.rsplit("_", 1)[0] + "_1"
    index.reset_mock()
    index.list.return_value = [[retained_id, stale_id]]
    endpoint(provider, request)
    index.delete.assert_called_once_with(ids=[stale_id], namespace=request.namespace)
    assert [call[0] for call in index.mock_calls] == ["list", "upsert", "delete"]


def test_long_text_retries_keep_ids_and_neighbor_links(provider):
    from app.configurations.config import CHUNK_THRESHOLD

    provider.file_processor._download_and_process_file.return_value = (
        "An actual sentence about this document. " * (CHUNK_THRESHOLD // 20 + 1),
        {"file_type": ".txt"},
    )
    request = request_for("https://example.com/file.txt")
    endpoint(provider, request)
    index = provider.pc.Index.return_value
    ids = [vector["id"] for vector in index.upsert.call_args.kwargs["vectors"]]
    assert len(ids) > 1
    index.list.return_value = [ids]
    endpoint(provider, request)
    vectors = [vector for call in index.upsert.call_args_list for vector in call.kwargs["vectors"]]
    assert set(vector["id"] for vector in vectors) == set(ids)
    for vector in vectors:
        for neighbor in ("prev_chunk_id", "next_chunk_id"):
            if neighbor in vector["metadata"]:
                assert vector["metadata"][neighbor] in ids
    index.delete.assert_not_called()


def test_identical_concurrent_retries_never_delete_new_chunks(provider):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    request = request_for("https://example.com/file.pdf")
    endpoint(provider, request)
    index = provider.pc.Index.return_value
    retained_id = index.upsert.call_args.kwargs["vectors"][0]["id"]
    stale_id = retained_id.rsplit("_", 1)[0] + "_1"
    index.list.return_value = [[retained_id, stale_id]]
    barrier = Barrier(2)
    index.upsert.side_effect = lambda **kwargs: barrier.wait(timeout=5)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(endpoint, provider, request) for _ in range(2)]
        responses = [future.result() for future in futures]
    assert all(response["indexed_file_urls"] == request.records[0].file_urls for response in responses)
    assert all(call.kwargs["ids"] == [stale_id] for call in index.delete.call_args_list)
