"""High-level agent operations shared by the MCP server and the CLI.

Keeping the logic here (rather than in server.py) means the MCP tools and the
CLI are thin wrappers over the same, testable functions.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from .config import Config, load_config
from .store import KnowledgeBase
from . import youtube

log = logging.getLogger("ytscholar.agent")

# Actionable guidance for the agent's most common real-world failure:
# YouTube unreachable (filtered network) or blocking the IP. Shared by the
# CLI and the MCP server so users always get the same fix.
NETWORK_HINT = (
    "YouTube was unreachable or refused the request. Fixes, in order:\n"
    "  1. If YouTube is filtered on your network, run your VPN/proxy and point\n"
    "     the agent at it, e.g.:\n"
    "       export YTSCHOLAR_HTTPS_PROXY=http://127.0.0.1:PORT\n"
    "     (socks5:// URLs work too if 'pip install pysocks' is installed).\n"
    "  2. If the error mentions IpBlocked or 'confirm you're not a bot',\n"
    "     pass your browser's YouTube cookies:\n"
    "       export YTSCHOLAR_COOKIES_FROM_BROWSER=firefox   # or chrome/brave/edge\n"
    "  3. See the 'Restricted-network install' section of the README."
)


class Agent:
    def __init__(self, config: Optional[Config] = None, kb: Optional[KnowledgeBase] = None):
        self.config = config or load_config()
        self.kb = kb or KnowledgeBase(self.config)

    # -- Feature 2: transcript for a single video --------------------------

    def transcript(
        self,
        video: str,
        languages: Optional[list[str]] = None,
        translate_to: Optional[str] = None,
        store: bool = True,
    ) -> dict:
        langs = languages or list(self.config.default_langs)
        t = youtube.get_transcript(
            video,
            languages=langs,
            translate_to=translate_to,
            proxies=self.config.proxies(),
            proxy_url=self.config.proxy_url(),
            cookies_from_browser=self.config.cookies_from_browser or None,
            cookies_file=self.config.cookies_file or None,
        )
        meta = youtube.VideoMeta(
            video_id=t.video_id,
            title=video,
            url=youtube.video_url(t.video_id),
        )
        stored_chunks = 0
        if store:
            # Enrich metadata cheaply via search is overkill; store as-is.
            stored_chunks = self.kb.add_transcript(meta, t, topic="")
        return {
            "video_id": t.video_id,
            "language": t.language_code,
            "is_generated": t.is_generated,
            "source": t.source,
            "num_snippets": len(t.snippets),
            "text": t.to_text(),
            "stored_chunks": stored_chunks,
        }

    # -- Feature 1: research a topic across the top videos -----------------

    def research_topic(
        self,
        topic: str,
        max_videos: int = 5,
        languages: Optional[list[str]] = None,
    ) -> dict:
        langs = languages or list(self.config.default_langs)
        # Governance: never exceed the hard cap regardless of what's requested.
        n = max(1, min(int(max_videos), self.config.max_videos_hard_cap))
        videos = youtube.search_videos(
            topic,
            limit=n,
            proxy=self.config.proxy_url(),
            cookies_from_browser=self.config.cookies_from_browser or None,
            cookies_file=self.config.cookies_file or None,
        )

        per_video = []
        total_new_chunks = 0
        for i, meta in enumerate(videos):
            entry = {
                "video_id": meta.video_id,
                "title": meta.title,
                "channel": meta.channel,
                "view_count": meta.view_count,
                "url": meta.url,
            }
            if self.kb.has_fresh_video(meta.video_id):
                entry["status"] = "cached"
                per_video.append(entry)
                continue
            try:
                t = youtube.get_transcript(
                    meta.video_id,
                    languages=langs,
                    proxies=self.config.proxies(),
                    proxy_url=self.config.proxy_url(),
                    cookies_from_browser=self.config.cookies_from_browser or None,
                    cookies_file=self.config.cookies_file or None,
                )
                chunks = self.kb.add_transcript(meta, t, topic=topic)
                total_new_chunks += chunks
                entry["status"] = "ingested"
                entry["language"] = t.language_code
                entry["chunks"] = chunks
            except youtube.TranscriptUnavailable as exc:
                entry["status"] = "no_transcript"
                entry["reason"] = str(exc)
            except Exception as exc:  # noqa: BLE001
                entry["status"] = "error"
                entry["reason"] = f"{type(exc).__name__}: {exc}"
            per_video.append(entry)
            if i < len(videos) - 1:
                time.sleep(self.config.request_delay_s)  # be polite to YouTube

        ingested = sum(1 for v in per_video if v.get("status") == "ingested")
        return {
            "topic": topic,
            "videos_considered": len(videos),
            "videos_ingested": ingested,
            "new_chunks": total_new_chunks,
            "videos": per_video,
            "knowledge": self.kb.stats(),
        }

    # -- Self-learning payoff: query the accumulated memory ----------------

    def search_knowledge(
        self, query: str, k: int = 5, topic: Optional[str] = None
    ) -> dict:
        hits = self.kb.search(query, k=k, topic=topic)
        return {
            "query": query,
            "topic": topic,
            "num_results": len(hits),
            "results": [h.to_dict() for h in hits],
        }

    # -- v0.2: evidence retrieval with source/channel awareness -------------

    def search_evidence(
        self, query: str, k: int = 6, topic: Optional[str] = None
    ) -> dict:
        """Evidence retrieval: same engine as search_knowledge, but grouped
        by video and channel so the consuming LLM can see how many sources
        actually back the evidence (and whether they concentrate in one
        channel). Deterministic; no LLM involved.
        """
        result = self.kb.search_evidence(query, k=k, topic=topic)
        return {"query": query, "topic": topic, **result}

    def stats(self) -> dict:
        return self.kb.stats()
