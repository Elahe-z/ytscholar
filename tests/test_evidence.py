"""Offline tests for evidence retrieval and source/channel awareness (v0.2)."""
from pathlib import Path

from ytscholar.config import Config
from ytscholar.store import KnowledgeBase
from ytscholar.youtube import Snippet, Transcript, VideoMeta, timestamped_url


def _kb() -> KnowledgeBase:
    # In-memory DB, embeddings off => pure FTS path, no heavy deps.
    cfg = Config(use_embeddings=False, chunk_chars=100)
    return KnowledgeBase(cfg, db_path=Path(":memory:"))


def _ingest(kb: KnowledgeBase, video_id: str, channel: str, text: str) -> None:
    kb.add_transcript(
        VideoMeta(video_id=video_id, title=f"t {video_id}", url="u", channel=channel),
        Transcript(
            video_id=video_id,
            language_code="en",
            is_generated=True,
            snippets=[Snippet(text=text, start=10.0)],
        ),
        topic="",
    )


# Test 1: several passages from several videos, all from ONE channel.


def test_all_from_one_channel_is_not_independent():
    kb = _kb()
    _ingest(kb, "vid00000001", "Chan A", "alpha kangaroo")
    _ingest(kb, "vid00000002", "Chan A", "alpha lemur")
    _ingest(kb, "vid00000003", "Chan A", "alpha wombat")
    res = kb.search_evidence("alpha", k=6)
    assert res["unique_channels"] == 1
    assert res["independent"] is False
    assert res["channel_distribution"] == {"Chan A": 3}
    assert res["warning"] and "Chan A" in res["warning"]
    assert len(res["videos"]) == 3
    assert len(res["passages"]) == 3


# Test 2: videos spread across channels — distribution must be exact.


def test_multiple_channels_distribution():
    kb = _kb()
    _ingest(kb, "vid00000011", "Chan A", "beta kangaroo")
    _ingest(kb, "vid00000012", "Chan B", "beta lemur")
    res = kb.search_evidence("beta", k=6)
    assert res["unique_channels"] == 2
    assert res["channel_distribution"] == {"Chan A": 1, "Chan B": 1}
    assert res["independent"] is True
    assert res["warning"] is None


# A dominant channel (strict majority) is flagged even with several channels.


def test_dominant_channel_flagged():
    kb = _kb()
    _ingest(kb, "vid00000021", "Chan A", "gamma one")
    _ingest(kb, "vid00000022", "Chan A", "gamma two")
    _ingest(kb, "vid00000023", "Chan A", "gamma three")
    _ingest(kb, "vid00000024", "Chan B", "gamma four")
    res = kb.search_evidence("gamma", k=8)
    assert res["unique_channels"] == 2
    assert res["channel_distribution"] == {"Chan A": 3, "Chan B": 1}
    assert res["independent"] is False
    assert res["warning"] and "3 of 4" in res["warning"]


# Test 3: one video contributing several passages must count as ONE source.


def test_video_counted_once_with_multiple_passages():
    kb = _kb()
    # Real transcripts are many short caption lines (chunking merges them up
    # to ~chunk_chars; it never splits a single line).
    kb.add_transcript(
        VideoMeta(video_id="vid00000031", title="t", url="u", channel="Chan A"),
        Transcript(
            video_id="vid00000031",
            language_code="en",
            is_generated=True,
            snippets=[
                Snippet(text=f"delta phrase number {i}", start=float(i))
                for i in range(40)
            ],  # -> several matching chunks from ONE video
        ),
        topic="",
    )
    _ingest(kb, "vid00000032", "Chan B", "delta other")
    res = kb.search_evidence("delta", k=10)
    vids = {v["video_id"]: v for v in res["videos"]}
    assert vids["vid00000031"]["passages"] > 1  # several passages...
    assert res["channel_distribution"] == {"Chan A": 1, "Chan B": 1}  # ...one source
    assert res["unique_channels"] == 2
    assert len(res["videos"]) == 2


# Tests 4 & 5: search_knowledge behavior unchanged; timestamped links intact.


def test_search_unchanged_and_links_intact():
    kb = _kb()
    _ingest(kb, "vid00000041", "Chan A", "epsilon proof here")
    # classic search still behaves exactly as before
    hits = kb.search("epsilon", k=5)
    assert hits and hits[0].video_id == "vid00000041"
    assert hits[0].link == timestamped_url("vid00000041", 10.0)
    # evidence is built on the same engine and keeps the same links
    res = kb.search_evidence("epsilon", k=5)
    assert res["passages"], "evidence should include the same hit"
    p = res["passages"][0]
    assert p["video_id"] == "vid00000041"
    assert p["start_seconds"] == 10.0
    assert p["link"] == timestamped_url("vid00000041", 10.0)
    assert "?t=" in p["link"]


def test_empty_result_is_clean():
    kb = _kb()
    _ingest(kb, "vid00000051", "Chan A", "zeta content")
    res = kb.search_evidence("unrelated words", k=5)
    assert res["passages"] == []
    assert res["videos"] == []
    assert res["unique_channels"] == 0
    assert res["warning"] is None


# Item: source-diversity figures describe the top-k window, not the whole KB.


def test_evidence_reports_match_coverage():
    kb = _kb()
    for i, ch in enumerate(["Chan A", "Chan B", "Chan C", "Chan D"]):
        _ingest(kb, f"vid0000006{i}", ch, "eta evidence")
    # k=2 can only see two of the four matching videos.
    res = kb.search_evidence("eta", k=2)
    assert res["k"] == 2
    assert len(res["videos"]) == 2
    assert res["total_matching_videos"] == 4
    assert res["warning"] and "4 videos match this query in total" in res["warning"]

    # With k wide enough, the window covers the match set and the caveat goes.
    full = kb.search_evidence("eta", k=10)
    assert full["total_matching_videos"] == 4
    assert len(full["videos"]) == 4
    assert full["warning"] is None
    assert full["independent"] is True


def test_coverage_caveat_appends_to_a_concentration_warning():
    kb = _kb()
    _ingest(kb, "vid00000071", "Chan A", "theta one")
    _ingest(kb, "vid00000072", "Chan A", "theta two")
    _ingest(kb, "vid00000073", "Chan B", "theta three")
    res = kb.search_evidence("theta", k=2)
    assert res["total_matching_videos"] == 3
    assert "same channel" in res["warning"]
    assert "3 videos match this query in total" in res["warning"]
