# ytscholar 🎓📺

**A local YouTube evidence layer (MVP).** It finds the top YouTube videos on
a topic, pulls their transcripts, and grows a local knowledge base — then
exposes searchable, source-aware **evidence** and **full transcripts** to LLM
clients, so the model can research, compare sources, and synthesize with
citations you can verify by clicking.

It runs entirely on your machine: no API keys, no cloud service, no
subscription. You can use it from the CLI or as an [MCP server](#mcp) inside
Claude Desktop, Cursor, Cline, or any MCP-compatible client.

> **Design boundary:** ytscholar collects, stores, and retrieves evidence.
> It does **not** reason — no claim extraction, no summarizing, no
> contradiction detection. Analysis and synthesis belong to the LLM client
> consuming this data.

> **Status: MVP.** The core loop — research → store → evidence retrieval with
> timestamped citations — works and is covered by offline tests (the network
> path was also validated manually against real YouTube). It is deliberately
> small. See [Limitations](#limitations) for what it is *not*.

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
4. **Retrieve evidence, source-aware.** The same retrieval, grouped by video
   and channel — so a model (or you) can see whether the evidence comes from
   genuinely different sources or from one channel repeated.

No YouTube Data API key is required. Search uses `yt-dlp`; transcripts use
`youtube-transcript-api` with a `yt-dlp` caption-download fallback.

## Why

Fetching one transcript is a solved problem — several tools do it. ytscholar's
value is the **accumulation** and the **evidence layer**: every ingest grows a
persistent local knowledge base, and retrieval returns evidence with precise,
clickable sources plus source-diversity analysis. An LLM client can search for
evidence, judge how independent the sources are, and pull a full transcript
when it needs complete context — then do the actual reasoning itself. That
makes ytscholar a small "research memory" you own: a SQLite file you can back
up, inspect, or delete.

## Architecture

Exactly what the code does today — ytscholar ends at evidence; the LLM client
does the analysis:

```
YouTube
  ↓
Video Discovery       yt-dlp ytsearch{N} (YouTube's own ranking);
                      hard cap per run (default 15); 30-day cache
  ↓
Full Transcript       youtube-transcript-api (primary)
                      → yt-dlp caption download + VTT parse (fallback)
  ↓
┌─────────────────────┬───────────────────────────────────────────┐
│ Full Transcript     │ Chunking (~900 chars; each chunk keeps    │
│ Storage             │ its start timestamp) → FTS5 index         │
└─────────────────────┴───────────────────────┬───────────────────┘
                                                ↓
Evidence Retrieval    passages + video/channel grouping
                      (FTS5 bm25; optional experimental re-rank)
                                                ↓
LLM client            analysis + synthesis            ← not part of
(Claude / GLM / …)    with verifiable citations       ytscholar
```

ytscholar never discards the full transcript: chunking and FTS5 exist for
*retrieval*; `get_transcript` always returns the complete stored text.

## Features

Only what exists and works today:

- 5 CLI commands: `research`, `transcript`, `search`, `evidence`, `stats`
- 5 MCP tools over the same core: `research_topic`, `get_transcript`,
  `search_knowledge`, `search_evidence`, `knowledge_stats`
- Keyword retrieval via SQLite FTS5 with bm25 ranking; optional topic filter
- **Evidence retrieval**: passages grouped by video and channel, with
  `unique_channels`, `channel_distribution`, and an independence warning
  (deterministic — channel variety only, no AI)
- Full transcripts kept in the DB and retrievable at any time
- Timestamped deep links on every search/evidence hit
- Per-video failure isolation — one broken video never kills a research run
- Politeness rails: hard per-run video cap, delay between requests, 30-day
  cache (no re-fetching what it already knows)
- Proxy and browser-cookie support for restricted networks (validated against
  a real filtered-network setup)
- Clean, actionable CLI errors instead of tracebacks
- Offline test suite (34 tests: URL/VTT parsing, chunking, storage, metadata
  merging, FTS retrieval, cache freshness, topic filter, evidence grouping,
  semantic re-rank plumbing) + CI on Python 3.10–3.12

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

# Evidence with source/channel analysis (human-readable):
ytscholar-cli evidence "how does RAG reduce hallucinations" --pretty

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
disabled by default; without it everything works via plain keyword search.
It is still **experimental**: the re-rank plumbing (vector packing, the cosine
ranking, and its fallbacks) is covered by tests that inject a stand-in
embedder, but no test exercises a real sentence-transformers model, and
retrieval quality with one has not been measured.

One behaviour to know about in a mixed knowledge base: chunks stored *before*
embeddings were enabled carry no vector, and the re-rank ranks only vectorised
chunks — so older material drops out of the ranking until it is re-ingested.
(If nothing in the candidate pool has a vector, search falls back to the
keyword ranking intact.)

Known search limitations (see also [Limitations](#limitations)): queries are
matched as OR-ed words (no quoted-phrase support), and there is no synonym
matching in keyword mode.

### Evidence retrieval (v0.2)

`ytscholar-cli evidence` / MCP `search_evidence` runs the **same** retrieval
engine, then groups the hits so source diversity is visible:

```json
{
  "passages": [
    {
      "video_id": "…", "title": "…", "channel": "…",
      "start_seconds": 763.1,
      "link": "https://youtu.be/…?t=763",
      "text": "…passage text…", "score": 7.85
    }
  ],
  "videos": [
    { "video_id": "…", "title": "…", "channel": "…", "url": "…", "passages": 3 }
  ],
  "unique_channels": 2,
  "channel_distribution": { "Channel A": 3, "Channel B": 1 },
  "total_matching_videos": 9,
  "k": 6,
  "independent": false,
  "warning": "3 of 4 videos behind these passages come from the same channel ('Channel A'). Evidence may not be fully independent. These figures describe the top 6 passages only: 9 videos match this query in total. Raise k to judge source diversity over the full match set."
}
```

The point: **N passages do not mean N sources.** If four matching videos come
from two channels — three of them from the same one — the model should know
that. `independent` is a deterministic channel-variety heuristic (`true` = no
single channel holds a strict majority of the videos behind the returned
passages); it makes no stronger epistemic claim.

**The diversity figures describe the top-`k` window, not the whole knowledge
base.** Grouping runs over the `k` passages actually returned, so a small `k`
can make a diverse match set look concentrated (or the reverse).
`total_matching_videos` reports how many videos match the query in total; when
it exceeds the number of videos listed, the `warning` says so and raising `k`
gives a fuller picture. Typical model workflow:
`search_evidence(query)` → judge sources → `get_transcript(video_id)` for any
source that needs full context → synthesize with citations.

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
five operations as the CLI, for use inside Claude Desktop, Cursor, Cline, …
(`research_topic`, `get_transcript`, `search_knowledge`, `search_evidence`,
`knowledge_stats`). Designed for the model workflow:
`research_topic` to ingest → `search_evidence` to find evidence and judge
source diversity → `get_transcript(video_id)` when full context is needed →
the model does the analysis and synthesis.

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
- **Semantic re-rank is experimental** — implemented and off by default. Its
  plumbing is tested with a stand-in embedder, but no real model is exercised
  and retrieval quality is unmeasured.
- **`independent` in evidence retrieval is a heuristic**: it reflects channel
  variety only (same-creator concentration), not true epistemic independence
  — two channels may still repeat the same primary source. It is also
  computed over the `k` returned passages, not the full match set; see
  `total_matching_videos`.
- `get_transcript` returns the **full** transcript text; for very long videos
  this is a large payload for an LLM context.
- A single-video `transcript` looks the real title/channel up via `yt-dlp`
  (one extra request, only when storing). If that lookup fails it stores the
  video with no title rather than guessing — and never overwrites metadata a
  previous richer ingest already recorded.
- The `topic` filter is an exact, case-sensitive match on the string passed
  to `research`.
- Ingest relies on scraping (`yt-dlp` / `youtube-transcript-api`): YouTube
  layout changes or IP blocks can break it. Cookies/proxy options mitigate.
- Video selection trusts YouTube's ranking as-is: no duration, language, or
  caption-availability filtering up front.

## Roadmap

Not implemented — kept deliberately out of this MVP:

- Claim extraction, contradiction detection, source lineage, evidence graphs
  (these are the LLM client's job; future versions may assist with them)
- Quoted-phrase queries and per-video diversity in search results
- Transcript length caps for LLM consumption
- Validate the embeddings path against a real model (quality, not just
  plumbing) and make semantic mode first-class
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
pytest -q          # 34 offline tests — no network, no heavy deps needed
```

CI (`.github/workflows/ci.yml`) runs this suite on Python 3.10–3.12 for every
push and pull request.

## License

MIT — see [LICENSE](LICENSE).
