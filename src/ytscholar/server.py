"""MCP server exposing the YouTube research agent.

Run over stdio (what Claude Desktop / Cursor / Cline launch):

    ytscholar

Or during development:

    python -m ytscholar.server

IMPORTANT: for a stdio MCP server, stdout is the protocol channel. All logging
must go to stderr, otherwise it corrupts the JSON-RPC stream.

IMPORTANT: tool functions must be ``async`` and offload the (synchronous,
network-bound) agent calls to a worker thread via ``anyio.to_thread``. The MCP
SDK executes synchronous tool functions directly on the server's event loop,
so one long research call would otherwise block the loop and every other tool
call — including light ones like knowledge_stats — would queue behind it
until the client times out (observed and reproduced with real clients).
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Optional

import anyio
from mcp.server.fastmcp import FastMCP

from .agent import Agent, NETWORK_HINT
from . import youtube

# Log to stderr ONLY — never stdout (that's the MCP transport).
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("ytscholar.server")

mcp = FastMCP("ytscholar")

# One shared agent for the process lifetime. Tools run in worker threads (see
# module docstring), so lazy init is guarded; the KnowledgeBase hands each
# thread its own SQLite connection.
_agent: Optional[Agent] = None
_agent_lock = threading.Lock()


def agent() -> Agent:
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = Agent()
                log.info("ytscholar agent ready. KB: %s", _agent.config.db_path)
    return _agent


def _tool_error(exc: Exception) -> dict:
    """Convert an exception into a structured tool result the client LLM can
    act on, instead of an opaque tool-error string."""
    out = {"error": f"{type(exc).__name__}: {exc}"}
    if isinstance(exc, youtube.TranscriptUnavailable) or "Network is unreachable" in str(exc):
        out["hint"] = NETWORK_HINT
    return out


async def _run(fn):
    """Run a sync agent call off the event loop and normalize its errors."""
    try:
        return await anyio.to_thread.run_sync(fn)
    except Exception as exc:  # noqa: BLE001 - tool boundary
        return _tool_error(exc)


@mcp.tool()
async def get_transcript(
    video: str,
    languages: str = "",
    translate_to: str = "",
    store: bool = True,
) -> dict:
    """Return the transcript/subtitles for a single YouTube video.

    Args:
        video: A YouTube URL or an 11-character video id.
        languages: Optional comma-separated preferred languages in priority
            order, e.g. "en" or "fa,en". Empty = server default.
        translate_to: Optional target language code to auto-translate the
            transcript into (uses YouTube's translation), e.g. "en".
        store: If true (default), also add this transcript to the local
            knowledge base so future searches can draw on it.

    Returns a dict with the plain-text transcript, language, and metadata.
    """
    langs = [s.strip() for s in languages.split(",") if s.strip()] or None
    a = agent()
    return await _run(lambda: a.transcript(
        video,
        languages=langs,
        translate_to=(translate_to or None),
        store=store,
    ))


@mcp.tool()
async def research_topic(topic: str, max_videos: int = 5, languages: str = "") -> dict:
    """Research a topic by mining the transcripts of the top YouTube videos.

    Searches YouTube for the topic, takes the top `max_videos` results, pulls
    each transcript, and ingests them into the agent's growing knowledge base.
    This is how the agent "learns" a subject. Use `search_knowledge` afterward
    to ask questions grounded in what was ingested.

    Note: on slow networks this can take minutes (one transcript fetch per
    video). Prefer small `max_videos` (1–3) there; the call keeps running
    server-side even if the client times out, and results still get ingested.

    Args:
        topic: The subject to research, e.g. "retrieval augmented generation".
        max_videos: How many top videos to mine (capped by server config).
        languages: Optional comma-separated preferred transcript languages.

    Returns a per-video ingestion report plus updated knowledge-base stats.
    """
    langs = [s.strip() for s in languages.split(",") if s.strip()] or None
    a = agent()
    return await _run(lambda: a.research_topic(
        topic, max_videos=max_videos, languages=langs
    ))


@mcp.tool()
async def search_knowledge(query: str, k: int = 5, topic: str = "") -> dict:
    """Semantic/keyword search over everything the agent has already learned.

    Retrieves the most relevant transcript passages from the local knowledge
    base, each with a deep link that opens the source video at the exact
    timestamp. Answer the user's question using these passages as evidence.

    Args:
        query: Natural-language question or keywords.
        k: Number of passages to return.
        topic: Optional filter to a topic previously passed to research_topic.
    """
    a = agent()
    return await _run(lambda: a.search_knowledge(query, k=k, topic=(topic or None)))


@mcp.tool()
async def search_evidence(query: str, k: int = 6, topic: str = "") -> dict:
    """Retrieve evidence passages relevant to a query from the stored
    transcripts, together with source analysis.

    Returns the matching passages (each with video_id, title, channel,
    start_seconds, text, score, and a timestamped link that opens the video
    at that exact moment), plus grouping over videos and channels:
    unique_channels and channel_distribution show how diverse the sources
    really are, and independent/warning flag when the evidence concentrates
    in a single channel. Several passages may come from the same video or
    channel — do not count them as independent sources.

    Those figures describe the k returned passages, not every stored match:
    total_matching_videos reports how many videos match the query overall, so
    when it exceeds the number of videos listed here, raise k before drawing
    conclusions about source diversity.

    Use this tool to find where the evidence is; use get_transcript(video_id)
    when you need the full transcript of a source for deeper analysis.

    Args:
        query: Natural-language question or keywords.
        k: Number of passages to return.
        topic: Optional filter to a topic previously passed to research_topic.
    """
    a = agent()
    return await _run(lambda: a.search_evidence(query, k=k, topic=(topic or None)))


@mcp.tool()
async def knowledge_stats() -> dict:
    """Report what the agent has learned so far: videos, chunks, topics, and
    whether semantic embeddings are active. Useful to check memory state."""
    a = agent()
    return await _run(lambda: a.stats())


def main() -> None:
    """Console-script entrypoint: run the MCP server over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
