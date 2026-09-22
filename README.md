# ytscholar 🎓📺

**A self-learning YouTube research agent (MVP).** It finds the top YouTube
videos on a topic, pulls their transcripts, and grows a local knowledge base
you can search — every result comes with a deep link to the exact second of
the source video.

It runs entirely on your machine: no API keys, no cloud service, no
subscription. You can use it from the CLI or as an [MCP server](#mcp) inside
Claude Desktop, Cursor, Cline, or any MCP-compatible client.

> **Status: MVP.** The core loop — research → store → search with timestamped
> citations — works and is covered by offline tests (the network path was also
> validated manually against real YouTube). It is deliberately small. See
> [Limitations](#limitations) for what it is *not*.

## What it does

1. **Research a topic.** Give it a subject; it finds the top YouTube videos
   for it, fetches their transcripts, and ingests them into a local knowledge
   base.
2. **Transcribe a link.** Give it a video URL/id; it returns the full
   transcript (with optional machine translation).
3. **Search what it has learned.** Ask a question; it retrieves the most
   relevant passages from *everything ever ingested*, each with a deep link
   that opens the source video at the right moment
   (`https://youtu.be/VIDEO_ID?t=SECONDS`).

No YouTube Data API key is required. Search uses `yt-dlp`; transcripts use
`youtube-transcript-api` with a `yt-dlp` caption-download fallback.

## Why

Fetching one transcript is a solved problem — several tools do it. ytscholar's
value is the **accumulation**: every ingest grows a persistent local knowledge
base, and every search runs over all of it, returning evidence with precise,
clickable sources. That makes it a small "research memory" you own — a SQLite
file you can back up, inspect, or delete — instead of a one-shot fetcher.

## Architecture

Exactly what the code does today:

```
YouTube Search        yt-dlp ytsearch{N} — mirrors YouTube's own ranking
        ↓
Video Selection       hard cap per run (default 15); 30-day cache skips
                      videos already learned
        ↓
Transcript Extraction youtube-transcript-api (primary)
                      → yt-dlp caption download + VTT parse (fallback)
        ↓
Chunking              ~900-char chunks; each keeps the start timestamp of
                      its first caption line
        ↓
Storage               SQLite (~/.ytscholar/knowledge.db): videos + chunks
                      + FTS5 full-text index
        ↓
Search / Retrieval    FTS5 bm25 keyword ranking over the whole KB
                      (optional semantic re-rank — experimental, see Search)
        ↓
Source + Timestamp    every hit returns youtu.be/ID?t=SECONDS
```

## Features

Only what exists and works today:

- 4 CLI commands: `research`, `transcript`, `search`, `stats`
- 4 MCP tools over the same core: `research_topic`, `get_transcript`,
  `search_knowledge`, `knowledge_stats`
- Keyword retrieval via SQLite FTS5 with bm25 ranking; optional topic filter
- Timestamped deep links on every search hit
- Per-video failure isolation — one broken video never kills a research run
- Politeness rails: hard per-run video cap, delay between requests, 30-day
  cache (no re-fetching what it already knows)
- Proxy and browser-cookie support for restricted networks (validated against
  a real filtered-network setup)
- Clean, actionable CLI errors instead of tracebacks
- Offline test suite (15 tests: URL/VTT parsing, chunking, storage, FTS
  retrieval, cache freshness, topic filter) + CI on Python 3.10–3.12

## Installation

The package is **not on PyPI yet** — install from source:

```bash
git clone https://github.com/Elahe-z/ytscholar
cd ytscholar
pip install -e .          # core: CLI + MCP server, keyword (FTS5) retrieval
```

Python 3.10+ is required. The optional `[embeddings]` extra is described in
[Search](#search).

## Configuration

All configuration is via environment variables (no config files, no secrets in
the repo):

| Variable | Default | Meaning |
|---|---|---|
| `YTSCHOLAR_HOME` | `~/.ytscholar` | Base dir for the knowledge DB |
| `YTSCHOLAR_DEFAULT_LANGS` | `en` | Preferred transcript languages, e.g. `fa,en` |
| `YTSCHOLAR_MAX_VIDEOS` | `15` | **Hard cap** on videos per research call |
| `YTSCHOLAR_REQUEST_DELAY` | `0.8` | Seconds between transcript fetches |
| `YTSCHOLAR_CACHE_TTL_DAYS` | `30` | Skip re-fetching a video seen within N days |
| `YTSCHOLAR_EMBEDDINGS` | `0` | `1` to enable semantic re-rank (experimental) |
| `YTSCHOLAR_EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `YTSCHOLAR_CHUNK_CHARS` | `900` | Approx chars per retrieval chunk |
| `YTSCHOLAR_HTTP_PROXY` | (from `HTTP_PROXY`) | Proxy for reaching YouTube |
| `YTSCHOLAR_HTTPS_PROXY` | (from `HTTPS_PROXY`) | HTTPS proxy for reaching YouTube |
| `YTSCHOLAR_COOKIES_FROM_BROWSER` | (unset) | Browser to read YouTube cookies from (`firefox`, `chrome`, `chromium`, `brave`, `edge`) |
| `YTSCHOLAR_COOKIES_FILE` | (unset) | Path to an exported `cookies.txt` |

## Usage

```bash
# Learn a topic from its top videos (English topics give the best ranking):
ytscholar-cli research "retrieval augmented generation" --max 5

# Ask questions about everything learned so far:
ytscholar-cli search "how does RAG reduce hallucinations"

# Get one video's transcript without storing it:
ytscholar-cli transcript "https://youtu.be/VIDEO_ID" --text-only --no-store

# What does the agent know?
ytscholar-cli stats
```

On a network where YouTube is filtered, point the agent at your proxy first —
see [Restricted networks](#restricted-networks-iran-and-similar-).

Tip: transcripts are usually English, so phrase `search` queries in English
for the best keyword matches.

## Search

Search is **keyword-based by default**:

1. The query is sanitized (alphanumeric tokens only — no FTS syntax
   injection is possible) and turned into an OR-query of its word tokens.
2. SQLite **FTS5** matches chunks with **bm25** ranking (lower = better) over
   the *entire* knowledge base, optionally filtered by topic (exact match).
3. A wider candidate pool (≥30) is fetched, ranked, and the top `k` returned.
4. Each hit carries `video_id`, `title`, `channel`, `start_seconds`, `text`,
   `score`, and a `link` deep link built from the chunk's stored start time.

**Optional semantic re-rank (experimental).** With
`pip install "ytscholar[embeddings]"` (pulls in `torch`, hundreds of MB) and
`YTSCHOLAR_EMBEDDINGS=1`, chunks are embedded at ingest time and query
vectors re-rank the FTS candidates by cosine similarity. This path is
implemented but **experimental: it is not covered by the test suite** and is
disabled by default. Without it, everything works via plain keyword search.

Known search limitations (see also [Limitations](#limitations)): queries are
matched as OR-ed words (no quoted-phrase support), and there is no synonym
matching in keyword mode.

## Research

`research_topic(topic, max_videos)` does exactly this, in order:

1. Clamps the video count to `min(max_videos, YTSCHOLAR_MAX_VIDEOS)`.
2. Searches YouTube via `yt-dlp` (`ytsearchN`, flat metadata) — the order is
   YouTube's own relevance ranking.
3. For each result: if the video was fetched within the cache TTL
   (default 30 days), it is marked `cached` and skipped — no re-download.
4. Otherwise the transcript is fetched (primary API, then yt-dlp fallback),
   chunked (~900 chars, timestamps preserved), and stored under that topic.
5. A short polite delay runs between videos.
6. One video failing (no captions, network error) only marks that video
   `no_transcript` / `error` with the reason — the run continues.
7. Returns a per-video report plus overall knowledge-base stats.

Statuses you will see: `ingested`, `cached`, `no_transcript`, `error`.

## Storage / Memory

- Single SQLite database: `~/.ytscholar/knowledge.db` (override the location
  with `YTSCHOLAR_HOME`).
- Tables: `videos` (metadata + full transcript text + `fetched_at`) and
  `chunks` (text, start time, optional embedding), plus an FTS5 full-text
  index kept in sync by triggers. WAL mode for safe concurrent reads.
- The DB **is** the agent's memory: it persists across sessions, it is safe to
  copy/back up, and deleting it resets what the agent knows.
- It lives outside the repository — no personal data ships with the code.

## MCP

An MCP server over stdio ships with the package (`ytscholar` command). Same
four operations as the CLI, for use inside Claude Desktop, Cursor, Cline, …

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "ytscholar": {
      "command": "ytscholar",
      "env": {
        "YTSCHOLAR_DEFAULT_LANGS": "en",
        "YTSCHOLAR_HTTPS_PROXY": "http://127.0.0.1:12334"
      }
    }
  }
}
```

Cursor (`~/.cursor/mcp.json`):

```json
{ "mcpServers": { "ytscholar": { "command": "ytscholar" } } }
```

If `ytscholar` is not on the client's PATH, use the absolute path
(`which ytscholar`) or `"command": "python", "args": ["-m", "ytscholar.server"]`.

Example prompts once connected: *"Research 'retrieval augmented generation'
from the top 5 YouTube videos"*, then *"From what you've learned, how does
re-ranking improve RAG?"* — answers come with timestamped citations.

## Limitations

Honest list for this MVP:

- **Keyword search only** (by default): OR-ed word tokens, no phrase
  support, no synonyms. English queries against English transcripts work
  well; Persian queries won't match English content.
- **Semantic re-rank is experimental** — implemented, off by default, not
  covered by tests.
- `get_transcript` returns the **full** transcript text; for very long videos
  this is a large payload for an LLM context.
- A single-video `transcript` stores the URL you passed as the video *title*
  in the knowledge base (real title enrichment is not implemented).
- The `topic` filter is an exact, case-sensitive match on the string passed
  to `research`.
- Ingest relies on scraping (`yt-dlp` / `youtube-transcript-api`): YouTube
  layout changes or IP blocks can break it. Cookies/proxy options mitigate.
- Video selection trusts YouTube's ranking as-is: no duration, language, or
  caption-availability filtering up front.

## Roadmap

Not implemented — kept deliberately out of this MVP:

- Quoted-phrase queries and per-video diversity in search results
- Real title/metadata enrichment for single-video transcripts; transcript
  length caps for LLM consumption
- Validate and test the embeddings path; make semantic mode first-class
- Optional tiny HTTP API over the same core, for workflow tools (n8n etc.)
- Publish to PyPI (`pip install ytscholar`)

## Restricted networks (Iran and similar)

If `pip install` fails with `No matching distribution found`, your network is
blocking pypi.org. Point pip at a local mirror:

```bash
pip install -e . -i https://mirror-pypi.runflare.com/simple/ \
    --trusted-host mirror-pypi.runflare.com
```

To reach YouTube itself, run your VPN/proxy and point the agent at it:

```bash
export YTSCHOLAR_HTTPS_PROXY="http://127.0.0.1:PORT"
ytscholar-cli transcript "https://youtu.be/VIDEO_ID" --text-only
```

> **Hiddify users:** the local mixed (HTTP+SOCKS) port is **12334** once the
> core is connected, so:
> `export YTSCHOLAR_HTTPS_PROXY=http://127.0.0.1:12334`
> (Ports like `17078` belong to the app itself, not the proxy — they refuse
> connections.)

If YouTube answers `IpBlocked` / "Sign in to confirm you're not a bot"
(common on VPN/datacenter IPs), pass your browser's cookies:

```bash
export YTSCHOLAR_COOKIES_FROM_BROWSER="firefox"   # or chrome / brave / edge
```

Or export a `cookies.txt` (browser extension) and set
`YTSCHOLAR_COOKIES_FILE=/path/to/cookies.txt`.

## Development

```bash
pip install -e ".[dev]"
pytest -q          # offline tests only — no network needed
```

CI (`.github/workflows/ci.yml`) runs this suite on Python 3.10–3.12 for every
push and pull request.

## License

MIT — see [LICENSE](LICENSE).
