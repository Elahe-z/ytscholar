"""The knowledge base — this is what makes the agent "self-learning".

Every transcript the agent ingests is chunked and stored in a local SQLite
database. Over time the KB grows, and ``search`` performs retrieval over
*everything the agent has ever seen*, not just the current video. This is a
RAG memory, not model fine-tuning: cheap, transparent, and inspectable.

Retrieval is hybrid:
  * Always: SQLite FTS5 keyword search (no extra dependencies).
  * If embeddings are enabled and available: a semantic re-rank on top,
    using cosine similarity over sentence-transformer vectors.

The DB is the single source of truth and is safe to copy, back up, or delete.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import Config
from .youtube import Snippet, Transcript, VideoMeta, timestamped_url, video_url

log = logging.getLogger("ytscholar.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    title        TEXT,
    channel      TEXT,
    url          TEXT,
    topic        TEXT,
    language     TEXT,
    duration     INTEGER,
    view_count   INTEGER,
    is_generated INTEGER,
    fetched_at   REAL,
    transcript   TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    start       REAL NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_video ON chunks(video_id);

-- Full-text index over chunk text. external-content table keyed to chunks.id.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
"""


@dataclass
class SearchHit:
    video_id: str
    title: str
    channel: str
    start: float
    text: str
    score: float

    @property
    def link(self) -> str:
        return timestamped_url(self.video_id, self.start)

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "channel": self.channel,
            "start_seconds": round(self.start, 1),
            "link": self.link,
            "text": self.text,
            "score": round(self.score, 4),
        }


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_snippets(snippets: list[Snippet], max_chars: int) -> list[Snippet]:
    """Merge fine-grained caption lines into ~max_chars retrieval chunks.

    Each output chunk keeps the ``start`` of its first line, so we can deep-link
    back to the exact moment in the video.
    """
    chunks: list[Snippet] = []
    buf: list[str] = []
    buf_start: Optional[float] = None
    buf_len = 0
    for sn in snippets:
        text = sn.text.strip()
        if not text:
            continue
        if buf_start is None:
            buf_start = sn.start
        buf.append(text)
        buf_len += len(text) + 1
        if buf_len >= max_chars:
            chunks.append(Snippet(text=" ".join(buf), start=buf_start))
            buf, buf_start, buf_len = [], None, 0
    if buf and buf_start is not None:
        chunks.append(Snippet(text=" ".join(buf), start=buf_start))
    return chunks


# ---------------------------------------------------------------------------
# Optional embedding backend (lazy, graceful)
# ---------------------------------------------------------------------------


class _Embedder:
    """Thin wrapper around sentence-transformers, loaded lazily.

    If the optional dependency is missing, ``available`` is False and the store
    silently falls back to keyword-only retrieval.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self.available = False
        try:  # keep import cost + failure local
            import numpy  # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401

            self.available = True
        except Exception as exc:  # noqa: BLE001
            log.info("embeddings disabled (%s). Falling back to FTS only.", exc)

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            log.info("loading embedding model %s ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]):
        model = self._ensure()
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def _pack(vec) -> bytes:
    return struct.pack(f"<{len(vec)}f", *[float(x) for x in vec])


def _unpack(blob: bytes):
    import numpy as np

    n = len(blob) // 4
    return np.array(struct.unpack(f"<{n}f", blob), dtype="float32")


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


class KnowledgeBase:
    def __init__(self, config: Config, db_path: Optional[Path] = None):
        self.config = config
        self.db_path = Path(db_path) if db_path else config.db_path
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        self._embedder = (
            _Embedder(config.embed_model) if config.use_embeddings else None
        )

    def close(self) -> None:
        self.conn.close()

    # -- ingest ------------------------------------------------------------

    def has_fresh_video(self, video_id: str) -> bool:
        """True if we already stored this video within the cache TTL."""
        row = self.conn.execute(
            "SELECT fetched_at FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        if not row:
            return False
        age_days = (time.time() - float(row["fetched_at"])) / 86400.0
        return age_days <= self.config.cache_ttl_days

    def add_transcript(
        self, meta: VideoMeta, transcript: Transcript, topic: str = ""
    ) -> int:
        """Store (or replace) a video + its chunked transcript. Returns #chunks."""
        full_text = transcript.to_text()
        chunks = chunk_snippets(transcript.snippets, self.config.chunk_chars)

        embeddings = None
        if self._embedder and self._embedder.available and chunks:
            try:
                embeddings = self._embedder.encode([c.text for c in chunks])
            except Exception as exc:  # noqa: BLE001
                log.warning("embedding failed, storing without vectors: %s", exc)

        cur = self.conn.cursor()
        cur.execute("DELETE FROM videos WHERE video_id = ?", (meta.video_id,))
        cur.execute("DELETE FROM chunks WHERE video_id = ?", (meta.video_id,))
        cur.execute(
            """INSERT INTO videos
               (video_id, title, channel, url, topic, language, duration,
                view_count, is_generated, fetched_at, transcript)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                meta.video_id,
                meta.title,
                meta.channel,
                meta.url,
                topic,
                transcript.language_code,
                meta.duration,
                meta.view_count,
                int(transcript.is_generated),
                time.time(),
                full_text,
            ),
        )
        for i, ch in enumerate(chunks):
            blob = None
            if embeddings is not None:
                blob = _pack(embeddings[i])
            cur.execute(
                "INSERT INTO chunks (video_id, chunk_index, start, text, embedding)"
                " VALUES (?,?,?,?,?)",
                (meta.video_id, i, ch.start, ch.text, blob),
            )
        self.conn.commit()
        return len(chunks)

    # -- retrieve ----------------------------------------------------------

    @staticmethod
    def _fts_query(query: str) -> str:
        # Turn a free-text query into a safe FTS5 OR-query of its word tokens.
        tokens = [t for t in "".join(
            c if c.isalnum() else " " for c in query
        ).split() if t]
        if not tokens:
            return '""'
        return " OR ".join(tokens)

    def search(
        self, query: str, k: int = 5, topic: Optional[str] = None
    ) -> list[SearchHit]:
        """Hybrid retrieval over the whole KB.

        Keyword recall via FTS5 (wide net), then optional semantic re-rank.
        """
        fts = self._fts_query(query)
        # Pull a wider candidate pool so the semantic re-rank has room to work.
        pool = max(k * 6, 30)
        sql = """
            SELECT c.id, c.video_id, c.start, c.text, c.embedding,
                   v.title, v.channel, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN videos v ON v.video_id = c.video_id
            WHERE chunks_fts MATCH ?
        """
        params: list = [fts]
        if topic:
            sql += " AND v.topic = ?"
            params.append(topic)
        sql += " ORDER BY rank LIMIT ?"
        params.append(pool)
        rows = self.conn.execute(sql, params).fetchall()

        if not rows:
            return []

        # Semantic re-rank if we can.
        if self._embedder and self._embedder.available:
            try:
                import numpy as np

                qv = self._embedder.encode([query])[0]
                scored = []
                for r in rows:
                    if r["embedding"] is None:
                        continue
                    v = _unpack(r["embedding"])
                    sim = float(np.dot(qv, v))  # both normalized => cosine
                    scored.append((sim, r))
                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    return [
                        SearchHit(
                            video_id=r["video_id"],
                            title=r["title"],
                            channel=r["channel"] or "",
                            start=r["start"],
                            text=r["text"],
                            score=sim,
                        )
                        for sim, r in scored[:k]
                    ]
            except Exception as exc:  # noqa: BLE001
                log.info("semantic re-rank skipped: %s", exc)

        # Keyword-only path. bm25 is ascending (lower = better); invert to score.
        hits = [
            SearchHit(
                video_id=r["video_id"],
                title=r["title"],
                channel=r["channel"] or "",
                start=r["start"],
                text=r["text"],
                score=-float(r["rank"]),
            )
            for r in rows[:k]
        ]
        return hits

    # -- evidence retrieval (v0.2) ------------------------------------------

    def search_evidence(
        self, query: str, k: int = 6, topic: Optional[str] = None
    ) -> dict:
        """Retrieve evidence passages plus source/channel analysis.

        Wraps :meth:`search` (same FTS5 engine, unchanged) and groups the
        hits by video and channel, so a consuming LLM can judge how diverse
        the sources really are.

        ``independent`` is a deterministic channel-variety heuristic only:
        True means no single channel accounts for a strict majority of the
        matching videos. It is not an epistemic guarantee of independence.
        """
        hits = self.search(query, k=k, topic=topic)

        videos: dict[str, dict] = {}
        channel_videos: dict[str, set] = {}
        for h in hits:
            v = videos.get(h.video_id)
            if v is None:
                v = {
                    "video_id": h.video_id,
                    "title": h.title,
                    "channel": h.channel or "(unknown channel)",
                    "url": video_url(h.video_id),
                    "passages": 0,
                }
                videos[h.video_id] = v
            v["passages"] += 1
            channel_videos.setdefault(v["channel"], set()).add(h.video_id)

        channel_distribution = {c: len(ids) for c, ids in channel_videos.items()}
        unique_channels = len(channel_distribution)
        independent = False
        warning = None
        if videos:
            n_videos = len(videos)
            dominant_channel, dominant_n = max(
                channel_distribution.items(), key=lambda kv: kv[1]
            )
            if unique_channels == 1:
                independent = False
                warning = (
                    f"All {n_videos} matching videos come from the same channel "
                    f"('{dominant_channel}'). Evidence may not be independent."
                )
            elif dominant_n > n_videos - dominant_n:
                independent = False
                warning = (
                    f"{dominant_n} of {n_videos} matching videos come from the "
                    f"same channel ('{dominant_channel}'). Evidence may not be "
                    "fully independent."
                )
            else:
                independent = True

        return {
            "passages": [h.to_dict() for h in hits],
            "videos": list(videos.values()),
            "unique_channels": unique_channels,
            "channel_distribution": channel_distribution,
            "independent": independent,
            "warning": warning,
        }

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict:
        v = self.conn.execute("SELECT COUNT(*) AS n FROM videos").fetchone()["n"]
        c = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        topics = self.conn.execute(
            "SELECT topic, COUNT(*) AS n FROM videos WHERE topic != '' "
            "GROUP BY topic ORDER BY n DESC"
        ).fetchall()
        emb = self.conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()["n"]
        return {
            "videos": v,
            "chunks": c,
            "chunks_with_embeddings": emb,
            "embeddings_enabled": bool(self._embedder and self._embedder.available),
            "topics": [{"topic": r["topic"], "videos": r["n"]} for r in topics],
            "db_path": str(self.db_path),
        }

    def list_videos(self, topic: Optional[str] = None, limit: int = 100) -> list[dict]:
        sql = "SELECT video_id, title, channel, topic, view_count FROM videos"
        params: list = []
        if topic:
            sql += " WHERE topic = ?"
            params.append(topic)
        sql += " ORDER BY fetched_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]
