"""Offline tests for env-var driven configuration parsing."""
from ytscholar.config import Config
from ytscholar.youtube import _ytdlp_base_opts


def test_network_knob_defaults(monkeypatch):
    for v in ("YTSCHOLAR_SOCKET_TIMEOUT", "YTSCHOLAR_SLEEP_REQUESTS"):
        monkeypatch.delenv(v, raising=False)
    c = Config(use_embeddings=False)
    assert c.socket_timeout_s == 30.0
    assert c.sleep_requests_s == 1.0


def test_network_knob_overrides(monkeypatch):
    monkeypatch.setenv("YTSCHOLAR_SOCKET_TIMEOUT", "12.5")
    monkeypatch.setenv("YTSCHOLAR_SLEEP_REQUESTS", "2")
    c = Config(use_embeddings=False)
    assert c.socket_timeout_s == 12.5
    assert c.sleep_requests_s == 2.0


def test_ytdlp_opts_carry_timeout_and_sleep():
    opts = _ytdlp_base_opts(timeout=20.0, sleep=2.0)
    assert opts["socket_timeout"] == 20.0
    assert opts["sleep_interval_requests"] == 2.0
    assert opts["noplaylist"] is True
