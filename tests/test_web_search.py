from __future__ import annotations

import pytest

from tools.web_search import _is_blocked_ip, web_search


class _RaisingDDGS:
    def __init__(self, exc: Exception):
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def text(self, *args, **kwargs):
        raise self._exc


@pytest.mark.parametrize(
    "exc",
    [AttributeError("unexpected shape"), NotImplementedError()],
)
def test_web_search_wraps_ddgs_errors(monkeypatch, exc):
    monkeypatch.setattr(
        "tools.web_search.DDGS", lambda *a, **k: _RaisingDDGS(exc)
    )

    with pytest.raises(RuntimeError, match=type(exc).__name__):
        web_search.invoke({"query": "anything"})


@pytest.mark.parametrize(
    "ip",
    [
        "100.64.0.1",        # CGNAT / shared address space
        "100.127.255.255",
        "169.254.169.254",   # cloud metadata
        "127.0.0.1",
        "10.0.0.1",
        "::1",
    ],
)
def test_non_global_ips_are_blocked(ip):
    assert _is_blocked_ip(ip)


@pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"])
def test_public_ips_are_allowed(ip):
    assert not _is_blocked_ip(ip)
