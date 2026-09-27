"""Offline regression tests for the v0.2.2 stability fix.

Covers the two halves of the fix:
1. Per-thread SQLite connections — MCP tools run in worker threads, so the
   KnowledgeBase must be usable from several threads at once (previously a
   second thread touching the single shared connection would raise
   ProgrammingError).
2. The timeout-injecting requests session — youtube-transcript-api sends
   requests without any timeout; our session must inject one.
"""
import threading

from ytscholar.config import Config
from ytscholar.store import KnowledgeBase
from ytscholar.youtube import Snippet, Transcript, VideoMeta, _timeout_session


def _ingest(kb, video_id, channel, text):
    kb.add_transcript(
        VideoMeta(video_id=video_id, title=f"t {video_id}", url="u", channel=channel),
        Transcript(
            video_id=video_id,
            language_code="en",
            is_generated=True,
            snippets=[Snippet(text=text, start=0.0)],
        ),
        topic="",
    )


def test_file_db_shared_across_threads(tmp_path):
    """A file-backed KB written in one thread must be readable from another,
    and concurrent read/write from two threads must not raise."""
    cfg = Config(use_embeddings=False, chunk_chars=100)
    kb = KnowledgeBase(cfg, db_path=tmp_path / "shared.db")

    _ingest(kb, "vid00000001", "Chan A", "hello searchable world")

    results = {}

    def reader():
        results["hits"] = kb.search("hello", k=3)

    def writer():
        _ingest(kb, "vid00000002", "Chan B", "concurrent write while reading")

    t1, t2 = threading.Thread(target=reader), threading.Thread(target=writer)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not t1.is_alive() and not t2.is_alive(), "thread deadlocked"
    assert results["hits"], "reader thread found nothing"
    assert kb.stats()["videos"] == 2, "writer thread's ingest lost"
    kb.close()


def test_timeout_session_injects_default(monkeypatch):
    """Every request through the session must carry our default timeout,
    without overriding an explicitly provided one."""
    captured = {}

    def fake_request(self, method, url, **kwargs):
        captured.update(kwargs, method=method, url=url)
        return "RESP"

    monkeypatch.setattr("requests.Session.request", fake_request)
    s = _timeout_session(12.5)

    assert s.get("http://example/x") == "RESP"
    assert captured["timeout"] == 12.5

    s.get("http://example/x", timeout=5)
    assert captured["timeout"] == 5
