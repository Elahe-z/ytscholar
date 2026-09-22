"""Central configuration, driven entirely by environment variables.

Everything has a sane default so the agent works with zero configuration, but
each knob can be overridden — which matters because the same package runs on
the author's laptop and on a stranger's machine after ``pip install ytscholar``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _default_db_path() -> Path:
    # Store the knowledge base under the user's home by default, so it persists
    # and grows across sessions (that "growing" is the self-learning part).
    root = os.environ.get("YTSCHOLAR_HOME")
    base = Path(root) if root else Path.home() / ".ytscholar"
    return base / "knowledge.db"


@dataclass(frozen=True)
class Config:
    """Runtime configuration for the agent."""

    # --- Storage -----------------------------------------------------------
    db_path: Path = field(default_factory=_default_db_path)

    # --- Language ----------------------------------------------------------
    # Descending priority list, e.g. "en" or "fa,en". Persian speakers can set
    # YTSCHOLAR_DEFAULT_LANGS=fa,en to prefer Farsi transcripts.
    default_langs: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            s.strip()
            for s in os.environ.get("YTSCHOLAR_DEFAULT_LANGS", "en").split(",")
            if s.strip()
        )
    )

    # --- Governance / safety rails ----------------------------------------
    # Hard ceiling on how many videos a single research call may fetch. This is
    # the equivalent of a per-run cap: it protects against runaway scraping,
    # rate-limit bans, and surprise cost/time. Even if a caller asks for 500,
    # we never exceed this.
    max_videos_hard_cap: int = field(
        default_factory=lambda: _get_int("YTSCHOLAR_MAX_VIDEOS", 15)
    )
    # Polite delay (seconds) between per-video transcript requests, to avoid
    # hammering YouTube and getting the IP blocked.
    request_delay_s: float = field(
        default_factory=lambda: _get_float("YTSCHOLAR_REQUEST_DELAY", 0.8)
    )
    # Skip re-fetching a video we already stored within this many days (cache).
    cache_ttl_days: int = field(
        default_factory=lambda: _get_int("YTSCHOLAR_CACHE_TTL_DAYS", 30)
    )

    # --- RAG / embeddings --------------------------------------------------
    # Turn semantic embeddings on/off. Off => keyword-only (FTS5) retrieval,
    # which needs no heavy dependencies. On requires: pip install ytscholar[embeddings]
    use_embeddings: bool = field(
        default_factory=lambda: _get_bool("YTSCHOLAR_EMBEDDINGS", False)
    )
    embed_model: str = field(
        default_factory=lambda: os.environ.get(
            "YTSCHOLAR_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    # Approx characters per transcript chunk when indexing for retrieval.
    chunk_chars: int = field(
        default_factory=lambda: _get_int("YTSCHOLAR_CHUNK_CHARS", 900)
    )

    # --- Networking / proxy -----------------------------------------------
    # In regions where YouTube is filtered (e.g. Iran), route requests through
    # a proxy. Falls back to the standard HTTP_PROXY / HTTPS_PROXY env vars, so
    # existing system proxies work with zero extra config.
    http_proxy: str = field(
        default_factory=lambda: os.environ.get("YTSCHOLAR_HTTP_PROXY")
        or os.environ.get("HTTP_PROXY", "")
    )
    https_proxy: str = field(
        default_factory=lambda: os.environ.get("YTSCHOLAR_HTTPS_PROXY")
        or os.environ.get("HTTPS_PROXY", "")
    )

    def proxies(self) -> Optional[dict]:
        """Return a requests-style proxies dict, or None if unset."""
        if not self.http_proxy and not self.https_proxy:
            return None
        http = self.http_proxy or self.https_proxy
        https = self.https_proxy or self.http_proxy
        return {"http": http, "https": https}

    def proxy_url(self) -> Optional[str]:
        """Return a single proxy URL (for yt-dlp), or None if unset."""
        return self.https_proxy or self.http_proxy or None

    # --- Cookies (bypass YouTube's "confirm you're not a bot") -------------
    # Behind a VPN/datacenter IP, YouTube often demands proof you're a real
    # signed-in user. Passing your browser cookies solves this. Either name a
    # browser to read cookies from live, or point to an exported cookies.txt.
    cookies_from_browser: str = field(
        default_factory=lambda: os.environ.get("YTSCHOLAR_COOKIES_FROM_BROWSER", "")
    )
    cookies_file: str = field(
        default_factory=lambda: os.environ.get("YTSCHOLAR_COOKIES_FILE", "")
    )


def load_config() -> Config:
    """Build a Config from the current environment."""
    return Config()
