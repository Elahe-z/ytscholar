"""Offline tests for the knowledge base: ingest, chunk, and FTS retrieval."""
from ytscholar.config import Config
from ytscholar.store import KnowledgeBase, chunk_snippets
from ytscholar.youtube import Snippet, Transcript, VideoMeta


def _kb() -> KnowledgeBase:
    # In-memory DB, embeddings off => pure FTS path, no heavy deps.
    cfg = Config(use_embeddings=False, chunk_chars=40)
    from pathlib import Path

    kb = KnowledgeBase(cfg, db_path=Path(":memory:"))
    return kb


def test_chunking_groups_by_chars():
    snippets = [Snippet(text="word", start=float(i)) for i in range(20)]
    chunks = chunk_snippets(snippets, max_chars=40)
    assert len(chunks) > 1
    # each chunk keeps a start time from its first snippet
    assert chunks[0].start == 0.0
    assert all(c.text for c in chunks)


def test_ingest_and_search():
    kb = _kb()
    meta = VideoMeta(video_id="vid00000001", title="Intro to HNSW", url="u", channel="Ch")
    t = Transcript(
        video_id="vid00000001",
        language_code="en",
        is_generated=True,
        snippets=[
            Snippet(text="HNSW is a graph based vector index", start=10.0),
            Snippet(text="it enables fast approximate nearest neighbor search", start=15.0),
            Snippet(text="totally unrelated cooking content here", start=20.0),
        ],
    )
    n = kb.add_transcript(meta, t, topic="vector-db")
    assert n >= 1

    hits = kb.search("nearest neighbor vector index", k=3)
    assert hits, "expected at least one FTS hit"
    top = hits[0]
    assert top.video_id == "vid00000001"
    assert "youtu.be/vid00000001?t=" in top.link

    stats = kb.stats()
    assert stats["videos"] == 1
    assert stats["chunks"] == n


def test_cache_freshness():
    kb = _kb()
    meta = VideoMeta(video_id="vid00000002", title="t", url="u")
    t = Transcript(video_id="vid00000002", language_code="en", is_generated=True,
                   snippets=[Snippet(text="hello", start=0.0)])
    assert kb.has_fresh_video("vid00000002") is False
    kb.add_transcript(meta, t, topic="x")
    assert kb.has_fresh_video("vid00000002") is True


def test_topic_filter():
    kb = _kb()
    for vid, topic, text in [
        ("vid00000010", "a", "apples are red fruit"),
        ("vid00000011", "b", "apples are also green sometimes"),
    ]:
        kb.add_transcript(
            VideoMeta(video_id=vid, title="t", url="u"),
            Transcript(video_id=vid, language_code="en", is_generated=True,
                       snippets=[Snippet(text=text, start=0.0)]),
            topic=topic,
        )
    hits = kb.search("apples", k=5, topic="a")
    assert all(h.video_id == "vid00000010" for h in hits)


def test_reingest_keeps_richer_metadata():
    """A single-video `transcript` call knows only the id. Re-ingesting must
    not wipe the real title/channel/topic a research run already recorded."""
    kb = _kb()
    rich = VideoMeta(
        video_id="vid00000020",
        title="Intro to HNSW",
        url="https://www.youtube.com/watch?v=vid00000020",
        channel="Real Channel",
        duration=600,
        view_count=1234,
    )
    t = Transcript(video_id="vid00000020", language_code="en", is_generated=True,
                   snippets=[Snippet(text="first pass", start=0.0)])
    kb.add_transcript(rich, t, topic="vector-db")

    bare = VideoMeta(video_id="vid00000020", title="", url="", channel="")
    t2 = Transcript(video_id="vid00000020", language_code="en", is_generated=True,
                    snippets=[Snippet(text="second pass", start=0.0)])
    kb.add_transcript(bare, t2, topic="")

    row = kb.list_videos()[0]
    assert row["title"] == "Intro to HNSW"
    assert row["channel"] == "Real Channel"
    assert row["topic"] == "vector-db"
    assert row["view_count"] == 1234


def test_reingest_prefers_new_metadata_when_present():
    """Preservation must not freeze stale values: real new data always wins."""
    kb = _kb()
    t = Transcript(video_id="vid00000021", language_code="en", is_generated=True,
                   snippets=[Snippet(text="x", start=0.0)])
    kb.add_transcript(
        VideoMeta(video_id="vid00000021", title="Old", url="u", channel="Old Ch"),
        t, topic="old-topic",
    )
    kb.add_transcript(
        VideoMeta(video_id="vid00000021", title="New", url="u2", channel="New Ch"),
        t, topic="new-topic",
    )
    row = kb.list_videos()[0]
    assert row["title"] == "New"
    assert row["channel"] == "New Ch"
    assert row["topic"] == "new-topic"
