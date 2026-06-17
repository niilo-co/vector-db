from app.controllers.base_controller import upsert_agent_video_transcript, upsert_video_transcript
from app.models.models import AgentVideoTranscriptRequest, VideoTranscriptRequest


class FakeBackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, fn, *args, **kwargs):
        self.tasks.append((fn, args, kwargs))


def _request(namespace=None, metadata=None):
    return VideoTranscriptRequest(
        id="class-live/room/session",
        transcript_json_url="https://assets.niilo.co/class-live/room/session/transcripts/transcript.json",
        namespace=namespace,
        metadata=metadata or {},
    )


def test_upsert_video_transcript_uses_requested_namespace():
    background_tasks = FakeBackgroundTasks()

    response = upsert_video_transcript(
        _request(namespace="niilo_videos_secure"),
        background_tasks,
        vector_db_service=object(),
    )

    assert response.namespace == "niilo_videos_secure"
    assert background_tasks.tasks[0][1][2] == "niilo_videos_secure"


def test_upsert_video_transcript_defaults_to_legacy_namespace():
    background_tasks = FakeBackgroundTasks()

    response = upsert_video_transcript(
        _request(),
        background_tasks,
        vector_db_service=object(),
    )

    assert response.namespace == "niilo_db"
    assert background_tasks.tasks[0][1][2] == "niilo_db"


def test_upsert_agent_video_transcript_uses_requested_index_and_namespace():
    background_tasks = FakeBackgroundTasks()
    request = AgentVideoTranscriptRequest(
        id="agent-resource-live-123",
        transcript_json_url="https://assets.niilo.co/class-live/room/session/transcripts/transcript.json",
        hls_url="https://assets.niilo.co/class-live/room/session/hls/playlist.m3u8",
        index_name="dynamic-agents",
        namespace="dynamic-agents_deck-review",
        metadata={"owner_id": "user-1", "available": True},
    )

    response = upsert_agent_video_transcript(
        request,
        background_tasks,
        vector_db_service=object(),
    )

    assert response.index_name == "dynamic-agents"
    assert response.namespace == "dynamic-agents_deck-review"
    assert background_tasks.tasks[0][1][1] == request
