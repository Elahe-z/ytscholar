"""YouTube access layer: search top videos and fetch transcripts.

No YouTube Data API key required.
  * Search  -> yt-dlp's ``ytsearch`` (mirrors YouTube's own ranking).
  * Transcripts -> youtube-transcript-api (primary), yt-dlp caption download
    (fallback).

Everything here is pure I/O + parsing and is deliberately decoupled from MCP,
so it can be unit-tested and reused from the CLI.
"""
from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("ytscholar.youtube")

# ---------------------------------------------------------------------------
# Video id / URL handling
# ---------------------------------------------------------------------------

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Common YouTube URL shapes -> capture the 11-char id.
_URL_PATTERNS = [
    re.compile(r"(?:v=|/v/|/embed/|/shorts/|/live/|youtu\.be/)([A-Za-z0-9_-]{11})"),
    re.compile(r"^([A-Za-z0-9_-]{11})$"),
]


def extract_video_id(url_or_id: str) -> str:
    """Return the 11-character video id from a URL or a bare id.

    Raises ValueError if nothing that looks like an id can be found.
    """
    if not url_or_id or not isinstance(url_or_id, str):
        raise ValueError("empty video reference")
    candidate = url_or_id.strip()
    if _ID_RE.match(candidate):
        return candidate
    for pat in _URL_PATTERNS:
        m = pat.search(candidate)
        if m:
            return m.group(1)
    raise ValueError(f"could not extract a YouTube video id from: {url_or_id!r}")


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def timestamped_url(video_id: str, start_seconds: float) -> str:
    """Deep link that opens the video at a given moment."""
    return f"https://youtu.be/{video_id}?t={int(start_seconds)}"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VideoMeta:
    video_id: str
    title: str
    url: str
    channel: str = ""
    duration: Optional[int] = None       # seconds
    view_count: Optional[int] = None


@dataclass
class Snippet:
    text: str
    start: float          # seconds
    duration: float = 0.0


@dataclass
class Transcript:
    video_id: str
    language_code: str
    is_generated: bool
    snippets: list[Snippet] = field(default_factory=list)
    source: str = "youtube-transcript-api"

    def to_text(self) -> str:
        return " ".join(s.text.strip() for s in self.snippets if s.text.strip())


# ---------------------------------------------------------------------------
# Search (yt-dlp, no API key)
# ---------------------------------------------------------------------------


def _apply_cookie_opts(
    opts: dict,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
) -> None:
    """Add yt-dlp cookie options in place, to pass YouTube's bot check."""
    if cookies_file:
        opts["cookiefile"] = cookies_file
    elif cookies_from_browser:
        # yt-dlp expects a tuple: (browser, profile, keyring, container)
        opts["cookiesfrombrowser"] = (cookies_from_browser,)


def search_videos(
    query: str,
    limit: int = 5,
    proxy: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
) -> list[VideoMeta]:
    """Return up to ``limit`` top videos for ``query`` using yt-dlp search.

    Uses ``extract_flat`` so we only pull lightweight metadata (fast, and no
    per-video page loads). The order mirrors YouTube's relevance ranking.
    ``proxy`` optionally routes the request (e.g. where YouTube is filtered).
    """
    import yt_dlp  # imported lazily so import cost is only paid when used

    limit = max(1, int(limit))
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "default_search": "ytsearch",
        "noplaylist": True,
    }
    if proxy:
        opts["proxy"] = proxy
    _apply_cookie_opts(opts, cookies_from_browser, cookies_file)
    results: list[VideoMeta] = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    for entry in (info or {}).get("entries", []) or []:
        if not entry:
            continue
        vid = entry.get("id")
        if not vid:
            continue
        results.append(
            VideoMeta(
                video_id=vid,
                title=entry.get("title") or "(untitled)",
                url=entry.get("url") or video_url(vid),
                channel=entry.get("channel") or entry.get("uploader") or "",
                duration=entry.get("duration"),
                view_count=entry.get("view_count"),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Transcript fetch — primary path (youtube-transcript-api)
# ---------------------------------------------------------------------------


def _build_proxy_config(proxies: Optional[dict]):
    """Return a youtube-transcript-api proxy_config, or None.

    Uses GenericProxyConfig when available (v1.x). Returns None on any failure
    so the caller can fall back to the no-proxy path.
    """
    if not proxies:
        return None
    try:
        from youtube_transcript_api.proxies import GenericProxyConfig

        return GenericProxyConfig(
            http_url=proxies.get("http"),
            https_url=proxies.get("https"),
        )
    except Exception:  # noqa: BLE001 - older versions lack this module
        return None


def _fetch_via_api(
    video_id: str,
    languages: list[str],
    translate_to: Optional[str] = None,
    proxies: Optional[dict] = None,
) -> Transcript:
    """Fetch a transcript using youtube-transcript-api.

    Handles both the modern instance API (v1.x: ``.fetch`` / ``.list``) and the
    legacy classmethod API (<1.0: ``.get_transcript`` / ``.list_transcripts``),
    so the package keeps working across library versions. ``proxies`` (a
    requests-style dict) routes traffic where YouTube is filtered.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    proxy_config = _build_proxy_config(proxies)

    # --- Modern v1.x instance API -----------------------------------------
    if hasattr(YouTubeTranscriptApi, "fetch") or hasattr(
        YouTubeTranscriptApi(), "fetch"
    ):
        api = (
            YouTubeTranscriptApi(proxy_config=proxy_config)
            if proxy_config is not None
            else YouTubeTranscriptApi()
        )
        if translate_to:
            transcript_list = api.list(video_id)
            base = transcript_list.find_transcript(languages)
            fetched = base.translate(translate_to).fetch()
            lang_code = translate_to
            is_generated = getattr(base, "is_generated", True)
        else:
            fetched = api.fetch(video_id, languages=languages)
            lang_code = getattr(fetched, "language_code", languages[0])
            is_generated = getattr(fetched, "is_generated", True)
        raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
        snippets = [
            Snippet(
                text=item["text"],
                start=float(item.get("start", 0.0)),
                duration=float(item.get("duration", 0.0)),
            )
            for item in raw
        ]
        return Transcript(
            video_id=video_id,
            language_code=lang_code,
            is_generated=bool(is_generated),
            snippets=snippets,
        )

    # --- Legacy <1.0 classmethod API --------------------------------------
    legacy_kwargs = {"proxies": proxies} if proxies else {}
    if translate_to:
        tl = YouTubeTranscriptApi.list_transcripts(video_id, **legacy_kwargs)
        base = tl.find_transcript(languages)
        raw = base.translate(translate_to).fetch()
        lang_code = translate_to
        is_generated = getattr(base, "is_generated", True)
    else:
        raw = YouTubeTranscriptApi.get_transcript(
            video_id, languages=languages, **legacy_kwargs
        )
        lang_code = languages[0]
        is_generated = True
    snippets = [
        Snippet(
            text=item["text"],
            start=float(item.get("start", 0.0)),
            duration=float(item.get("duration", 0.0)),
        )
        for item in raw
    ]
    return Transcript(
        video_id=video_id,
        language_code=lang_code,
        is_generated=bool(is_generated),
        snippets=snippets,
    )


# ---------------------------------------------------------------------------
# Transcript fetch — fallback path (yt-dlp caption download + VTT parse)
# ---------------------------------------------------------------------------

# WEBVTT cue timestamp: 00:00:01.234 --> 00:00:03.456
_VTT_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
_VTT_TAGS = re.compile(r"<[^>]+>")


def _vtt_ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(text: str) -> list[Snippet]:
    """Parse a WEBVTT caption file into de-duplicated snippets.

    Auto-generated YouTube VTT is full of rolling/overlapping duplicate lines;
    we keep the first occurrence of each line and drop immediate repeats.
    """
    snippets: list[Snippet] = []
    current_start = 0.0
    current_end = 0.0
    seen_last = ""
    lines = text.splitlines()
    # Skip the WEBVTT header block ("WEBVTT", "Kind: captions", "Language:",
    # ...): per the spec it ends at the first blank line, before any cue.
    i = 0
    if lines and lines[0].upper().lstrip().startswith("WEBVTT"):
        while i < len(lines) and lines[i].strip():
            i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
    for line in lines[i:]:
        line = line.strip()
        if not line or line.isdigit():
            continue
        tm = _VTT_TIME.search(line)
        if tm:
            current_start = _vtt_ts_to_seconds(*tm.groups()[0:4])
            current_end = _vtt_ts_to_seconds(*tm.groups()[4:8])
            continue
        if "-->" in line:
            continue
        clean = _VTT_TAGS.sub("", line).strip()
        if not clean or clean == seen_last:
            continue
        seen_last = clean
        snippets.append(
            Snippet(text=clean, start=current_start, duration=max(0.0, current_end - current_start))
        )
    return snippets


def _fetch_via_ytdlp(
    video_id: str,
    languages: list[str],
    proxy: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
) -> Transcript:
    """Fallback: let yt-dlp download the caption track, then parse the VTT."""
    import yt_dlp

    with tempfile.TemporaryDirectory() as tmp:
        outtmpl = str(Path(tmp) / "%(id)s")
        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": languages + ["en"],
            "subtitlesformat": "vtt",
            "outtmpl": outtmpl,
            "noplaylist": True,
        }
        if proxy:
            opts["proxy"] = proxy
        _apply_cookie_opts(opts, cookies_from_browser, cookies_file)
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video_url(video_id)])
        vtts = sorted(Path(tmp).glob("*.vtt"))
        if not vtts:
            raise RuntimeError("yt-dlp found no caption track")
        # Prefer a requested language file if present.
        chosen = vtts[0]
        for lang in languages:
            match = [p for p in vtts if f".{lang}" in p.name]
            if match:
                chosen = match[0]
                break
        text = chosen.read_text(encoding="utf-8", errors="replace")
        lang_code = chosen.name.split(".")[-2] if "." in chosen.name else languages[0]
    snippets = parse_vtt(text)
    if not snippets:
        raise RuntimeError("caption track parsed to zero snippets")
    return Transcript(
        video_id=video_id,
        language_code=lang_code,
        is_generated=True,
        snippets=snippets,
        source="yt-dlp",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class TranscriptUnavailable(Exception):
    """Raised when no transcript can be obtained by any method."""


def get_transcript(
    video: str,
    languages: Optional[list[str]] = None,
    translate_to: Optional[str] = None,
    proxies: Optional[dict] = None,
    proxy_url: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    cookies_file: Optional[str] = None,
) -> Transcript:
    """Fetch a transcript for a video (URL or id), trying API then yt-dlp.

    ``proxies`` is a requests-style dict for youtube-transcript-api; ``proxy_url``
    is a single URL for the yt-dlp fallback. ``cookies_from_browser`` /
    ``cookies_file`` let the yt-dlp fallback pass YouTube's bot check. All
    optional (for filtered regions / VPN IPs).

    Raises TranscriptUnavailable with a human-readable reason on total failure.
    """
    video_id = extract_video_id(video)
    langs = languages or ["en"]
    errors: list[str] = []

    try:
        return _fetch_via_api(video_id, langs, translate_to=translate_to, proxies=proxies)
    except Exception as exc:  # noqa: BLE001 - we want to try the fallback
        errors.append(f"transcript-api: {type(exc).__name__}: {exc}")
        log.info("primary transcript fetch failed for %s (%s)", video_id, exc)

    if translate_to is None:  # yt-dlp fallback can't translate
        try:
            return _fetch_via_ytdlp(
                video_id,
                langs,
                proxy=proxy_url,
                cookies_from_browser=cookies_from_browser,
                cookies_file=cookies_file,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"yt-dlp: {type(exc).__name__}: {exc}")
            log.info("fallback transcript fetch failed for %s (%s)", video_id, exc)

    raise TranscriptUnavailable(
        f"No transcript available for {video_id}. Tried: " + " | ".join(errors)
    )
