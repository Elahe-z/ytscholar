"""Offline tests for URL/id parsing and VTT parsing (no network)."""
import pytest

from ytscholar.youtube import (
    _apply_cookie_opts,
    extract_video_id,
    parse_vtt,
    timestamped_url,
)


def test_cookie_opts_browser_and_profile():
    """Bare browser name -> default profile; "chrome:Profile 10" selects it."""
    bare, profiled, filed = {}, {}, {}
    _apply_cookie_opts(bare, cookies_from_browser="chrome")
    _apply_cookie_opts(profiled, cookies_from_browser="chrome:Profile 10")
    _apply_cookie_opts(filed, cookies_from_browser="chrome", cookies_file="/c.txt")
    assert bare["cookiesfrombrowser"] == ("chrome", None)
    assert profiled["cookiesfrombrowser"] == ("chrome", "Profile 10")
    # an explicit cookies file always wins
    assert "cookiesfrombrowser" not in filed
    assert filed["cookiefile"] == "/c.txt"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id(raw, expected):
    assert extract_video_id(raw) == expected


def test_extract_video_id_bad():
    with pytest.raises(ValueError):
        extract_video_id("not a youtube link")


def test_timestamped_url():
    assert timestamped_url("abc12345678", 90.7) == "https://youtu.be/abc12345678?t=90"


def test_parse_vtt_dedupes_and_times():
    vtt = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000
hello world

00:00:02.000 --> 00:00:04.000
hello world

00:00:04.000 --> 00:00:06.000
second line
"""
    snippets = parse_vtt(vtt)
    texts = [s.text for s in snippets]
    assert texts == ["hello world", "second line"]
    assert snippets[0].start == 0.0
    assert snippets[1].start == 4.0


def test_parse_vtt_skips_header_block():
    """Real YouTube VTT files carry a header block (Kind/Language) that must
    never leak into the transcript text."""
    vtt = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.000
first cue
"""
    snippets = parse_vtt(vtt)
    assert [s.text for s in snippets] == ["first cue"]
