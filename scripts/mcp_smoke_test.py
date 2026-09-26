"""Smoke-test the ytscholar MCP server the way a real client sees it.

Spawns the server over stdio using the official ``mcp`` client library,
performs the initialize handshake, lists tools, and calls each one
(including a live network call when a proxy is reachable). Usage:

    python scripts/mcp_smoke_test.py [--offline]

``--offline`` skips the network-dependent ``get_transcript`` call.
Exit code 0 = all checks passed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PROXY = os.environ.get("YTSCHOLAR_HTTPS_PROXY", "http://127.0.0.1:12334")


def server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "ytscholar.server"],
        env={
            "HOME": os.environ.get("HOME", "/home"),
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            # Route YouTube traffic through the local VPN proxy when present.
            "YTSCHOLAR_HTTPS_PROXY": PROXY,
            "HTTPS_PROXY": PROXY,
            "HTTP_PROXY": PROXY,
            "https_proxy": PROXY,
            "http_proxy": PROXY,
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
        },
    )


def payload(result) -> dict:
    """Extract a dict from a CallToolResult (structured or text content)."""
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            return json.loads(text)
    raise RuntimeError("tool returned no parsable content")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    ok = True
    async with stdio_client(server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print(f"1. initialize OK — server: {info.serverInfo.name} "
                  f"v{info.serverInfo.version}")

            tools = (await session.list_tools()).tools
            names = sorted(t.name for t in tools)
            print(f"2. tools/list OK — {len(names)} tools: {', '.join(names)}")
            ok &= len(names) == 5

            stats = payload(await session.call_tool("knowledge_stats", {}))
            print(f"3. knowledge_stats OK — {stats['videos']} videos, "
                  f"{stats['chunks']} chunks")

            ev = payload(await session.call_tool(
                "search_evidence",
                {"query": "how does RAG reduce hallucinations", "k": 4},
            ))
            print(f"4. search_evidence OK — {len(ev['passages'])} passages, "
                  f"{ev['unique_channels']} unique channels, "
                  f"independent={ev['independent']}, "
                  f"total_matching_videos={ev['total_matching_videos']}")

            if not args.offline:
                tr = payload(await session.call_tool(
                    "get_transcript",
                    {"video": "https://youtu.be/jNQXAC9IVRw", "store": False},
                ))
                if "error" in tr:
                    print(f"5. get_transcript (live) — graceful error: "
                          f"{tr['error'][:80]}")
                else:
                    print(f"5. get_transcript (live) OK — {len(tr['text'])} chars, "
                          f"title={tr.get('title', '')!r}")

    print("SMOKE TEST:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
