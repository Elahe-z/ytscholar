"""A small CLI so the agent is usable (and testable) without an MCP client.

    ytscholar-cli transcript "https://youtu.be/VIDEO"
    ytscholar-cli research "vector databases" --max 5
    ytscholar-cli search "how does HNSW indexing work"
    ytscholar-cli stats
"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import Agent, NETWORK_HINT
from . import youtube


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def _fail(msg: str, hint: str = "") -> int:
    """Print an error the way a CLI should: message + optional hint, no traceback."""
    print(f"error: {msg}", file=sys.stderr)
    if hint:
        print(f"\n{hint}", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ytscholar-cli",
        description="Self-learning YouTube research agent (local CLI).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_t = sub.add_parser("transcript", help="Get the transcript for one video")
    p_t.add_argument("video", help="YouTube URL or 11-char id")
    p_t.add_argument("--langs", default="", help="comma-separated, e.g. fa,en")
    p_t.add_argument("--translate-to", default="", help="target lang code")
    p_t.add_argument("--no-store", action="store_true", help="don't add to KB")
    p_t.add_argument("--text-only", action="store_true", help="print plain text")

    p_r = sub.add_parser("research", help="Mine transcripts of top videos on a topic")
    p_r.add_argument("topic")
    p_r.add_argument("--max", type=int, default=5, dest="max_videos")
    p_r.add_argument("--langs", default="")

    p_s = sub.add_parser("search", help="Query the learned knowledge base")
    p_s.add_argument("query")
    p_s.add_argument("-k", type=int, default=5)
    p_s.add_argument("--topic", default="")

    sub.add_parser("stats", help="Show knowledge-base statistics")

    args = parser.parse_args(argv)
    agent = Agent()

    if args.cmd == "transcript":
        langs = [s.strip() for s in args.langs.split(",") if s.strip()] or None
        try:
            res = agent.transcript(
                args.video,
                languages=langs,
                translate_to=(args.translate_to or None),
                store=not args.no_store,
            )
        except youtube.TranscriptUnavailable as exc:
            return _fail(str(exc), NETWORK_HINT)
        except Exception as exc:  # noqa: BLE001 - CLI boundary: fail cleanly
            return _fail(f"{type(exc).__name__}: {exc}")
        if args.text_only:
            print(res["text"])
        else:
            _print(res)
    elif args.cmd == "research":
        langs = [s.strip() for s in args.langs.split(",") if s.strip()] or None
        try:
            res = agent.research_topic(
                args.topic, max_videos=args.max_videos, languages=langs
            )
        except Exception as exc:  # noqa: BLE001 - search failure is network-ish
            return _fail(f"{type(exc).__name__}: {exc}", NETWORK_HINT)
        _print(res)
    elif args.cmd == "search":
        _print(agent.search_knowledge(args.query, k=args.k, topic=(args.topic or None)))
    elif args.cmd == "stats":
        _print(agent.stats())
    else:  # pragma: no cover
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
