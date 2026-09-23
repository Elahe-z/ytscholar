"""Offline tests for the semantic re-rank path.

The real embedder needs sentence-transformers (and torch). These tests inject a
tiny deterministic stand-in instead, so the *plumbing* — vector packing, the
cosine re-rank, and its failure handling — is covered on every CI run without
any heavy dependency.
"""
from pathlib import Path

from ytscholar.config import Config
from ytscholar.store import KnowledgeBase, _dot, _pack, _unpack
from ytscholar.youtube import Snippet, Transcript, VideoMeta


class FakeEmbedder:
    """Maps text to a unit vector by keyword, so similarity is predictable."""

    available = True

    def __init__(self):
        self.calls: list[list[str]] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        t = text.lower()
        # Two orthogonal unit axes: "cooking" and "vector databases".
        return [1.0, 0.0] if "recipe" in t else [0.0, 1.0]

    def encode(self, texts: list[str]):
        self.calls.append(list(texts))
        return [self._vector(t) for t in texts]


class BrokenEmbedder:
    available = True

    def encode(self, texts: list[str]):
        raise RuntimeError("model failed to load")


def _kb(embedder=None) -> KnowledgeBase:
    cfg = Config(use_embeddings=False, chunk_chars=100)
    return KnowledgeBase(cfg, db_path=Path(":memory:"), embedder=embedder)


def _ingest(kb: KnowledgeBase, video_id: str, text: str) -> None:
    kb.add_transcript(
        VideoMeta(video_id=video_id, title=f"t {video_id}", url="u", channel="Ch"),
        Transcript(
            video_id=video_id,
            language_code="en",
            is_generated=True,
            snippets=[Snippet(text=text, start=5.0)],
        ),
        topic="",
    )


def test_pack_unpack_roundtrip():
    vec = [0.5, -0.25, 0.125, 0.0]
    assert list(_unpack(_pack(vec))) == vec


def test_unpack_needs_no_numpy():
    """The retrieval path must work on a bare install (core deps only)."""
    blob = _pack([0.1] * 384)
    out = _unpack(blob)
    assert isinstance(out, tuple) and len(out) == 384


def test_dot_is_cosine_for_unit_vectors():
    assert _dot([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _dot([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_ingest_stores_vectors():
    kb = _kb(FakeEmbedder())
    _ingest(kb, "vid00000001", "a recipe for bread")
    stats = kb.stats()
    assert stats["embeddings_enabled"] is True
    assert stats["chunks_with_embeddings"] == stats["chunks"] > 0


def test_semantic_rerank_beats_keyword_order():
    """Both docs match the query keywords, but only one is semantically right;
    the re-rank must surface it first."""
    kb = _kb(FakeEmbedder())
    # Keyword-wise the cooking doc repeats "index" more, so bm25 favours it.
    _ingest(kb, "vid00000011", "recipe index index index index index")
    _ingest(kb, "vid00000012", "the index of a vector database")
    hits = kb.search("index", k=2)
    assert [h.video_id for h in hits] == ["vid00000012", "vid00000011"]
    # Scores are cosine similarities from the injected embedder.
    assert hits[0].score == 1.0
    assert hits[1].score == 0.0


def test_rerank_falls_back_when_no_candidate_has_a_vector():
    """Content ingested before embeddings were switched on has no vector. The
    re-rank must not swallow it — search falls back to the keyword ranking."""
    kb = _kb()  # no embedder: stores plain chunks
    _ingest(kb, "vid00000021", "vector database index")
    kb._embedder = FakeEmbedder()  # embeddings enabled later
    hits = kb.search("index", k=5)
    assert hits and hits[0].video_id == "vid00000021"


def test_rerank_ranks_only_vectorised_chunks():
    """Documented consequence of a mixed KB: once the re-rank has at least one
    vectorised candidate, un-vectorised ones drop out of that ranking. A
    re-ingest (or a fresh research run) restores them."""
    kb = _kb()
    _ingest(kb, "vid00000022", "vector database index")  # no vector
    kb._embedder = FakeEmbedder()
    _ingest(kb, "vid00000023", "another index of a vector database")  # vector
    hits = kb.search("index", k=5)
    assert [h.video_id for h in hits] == ["vid00000023"]


def test_broken_embedder_falls_back_to_keyword_search():
    kb = _kb(FakeEmbedder())
    _ingest(kb, "vid00000031", "vector database index")
    kb._embedder = BrokenEmbedder()
    hits = kb.search("index", k=5)
    assert hits and hits[0].video_id == "vid00000031"


def test_embedding_failure_at_ingest_still_stores_chunks():
    kb = _kb(BrokenEmbedder())
    _ingest(kb, "vid00000041", "vector database index")
    stats = kb.stats()
    assert stats["chunks"] > 0
    assert stats["chunks_with_embeddings"] == 0
    assert kb.search("index", k=5)  # still retrievable via FTS
