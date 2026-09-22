from __future__ import annotations

import socket

import pytest
import requests

from tools.web_search import (
    MAX_CHARS,
    _is_blocked_ip,
    _resolve_and_validate,
    _snippet_fallback,
    fetch_page,
    web_search,
)


class _RaisingDDGS:
    def __init__(self, exc: Exception):
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def text(self, *args, **kwargs):
        raise self._exc


class _ResultsDDGS:
    def __init__(self, results):
        self._results = results

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def text(self, *args, **kwargs):
        return self._results


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


def test_web_search_rejects_empty_query():
    with pytest.raises(ValueError):
        web_search.invoke({"query": "   "})


def test_web_search_maps_result_fields(monkeypatch):
    monkeypatch.setattr(
        "tools.web_search.DDGS",
        lambda *a, **k: _ResultsDDGS(
            [{"title": "T", "href": "https://e.com", "body": "S"}]
        ),
    )

    results = web_search.invoke({"query": "x"})

    assert results == [{"title": "T", "url": "https://e.com", "snippet": "S"}]


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


def _addr(ip: str):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


def test_resolve_and_validate_rejects_bad_scheme():
    with pytest.raises(ValueError, match="http/https"):
        _resolve_and_validate("ftp://example.com/file")


def test_resolve_and_validate_rejects_missing_hostname():
    with pytest.raises(ValueError, match="hostname"):
        _resolve_and_validate("http://")


def test_resolve_and_validate_rejects_disallowed_port():
    with pytest.raises(ValueError, match="Port"):
        _resolve_and_validate("http://example.com:8080/")


def test_resolve_and_validate_rejects_blocked_ip(monkeypatch):
    monkeypatch.setattr("tools.web_search.socket.getaddrinfo", lambda *a, **k: _addr("127.0.0.1"))
    with pytest.raises(ValueError, match="blocked address"):
        _resolve_and_validate("http://example.com/")


def test_resolve_and_validate_rejects_unresolvable_host(monkeypatch):
    def _boom(*_a, **_k):
        raise socket.gaierror("no such host")

    monkeypatch.setattr("tools.web_search.socket.getaddrinfo", _boom)
    with pytest.raises(ValueError, match="Could not resolve"):
        _resolve_and_validate("http://nope.invalid/")


def test_resolve_and_validate_returns_public_ip(monkeypatch):
    monkeypatch.setattr(
        "tools.web_search.socket.getaddrinfo", lambda *a, **k: _addr("93.184.216.34")
    )
    assert _resolve_and_validate("https://example.com/") == "93.184.216.34"


def test_snippet_fallback_without_snippet_raises():
    with pytest.raises(RuntimeError, match="no search snippet"):
        _snippet_fallback("https://e.com", "  ", "HTTP 403")


def test_snippet_fallback_labels_reason_and_source():
    text = _snippet_fallback("https://e.com", "snippet text", "HTTP 403")
    assert "HTTP 403" in text
    assert "https://e.com" in text
    assert "snippet text" in text


class _Resp:
    def __init__(
        self,
        *,
        status_code=200,
        headers=None,
        body=b"",
        encoding="utf-8",
        redirect=False,
        permanent=False,
        raise_exc=None,
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body
        self.encoding = encoding
        self.is_redirect = redirect
        self.is_permanent_redirect = permanent
        self._raise_exc = raise_exc
        self.closed = False

    def raise_for_status(self):
        if self._raise_exc is not None:
            raise self._raise_exc

    def iter_content(self, chunk_size=65536):
        yield self._body

    def close(self):
        self.closed = True


def _patch_fetch(monkeypatch, responses):
    monkeypatch.setattr(
        "tools.web_search._resolve_and_validate", lambda url: "93.184.216.34"
    )
    if isinstance(responses, list):
        it = iter(responses)

        def _get(*_a, **_k):
            return next(it)

        monkeypatch.setattr("tools.web_search.requests.get", _get)
    else:
        monkeypatch.setattr("tools.web_search.requests.get", lambda *a, **k: responses)


def test_fetch_page_returns_clean_text(monkeypatch):
    body = b"<html><body><h1>Hello</h1><script>bad()</script><p>World</p></body></html>"
    _patch_fetch(
        monkeypatch,
        _Resp(headers={"Content-Type": "text/html; charset=utf-8"}, body=body),
    )

    text = fetch_page.invoke({"url": "https://e.com", "snippet": ""})

    assert "Hello" in text
    assert "World" in text
    assert "bad()" not in text


def test_fetch_page_http_error_falls_back_to_snippet(monkeypatch):
    err = requests.HTTPError("403")
    err.response = _Resp(status_code=403)
    _patch_fetch(
        monkeypatch,
        _Resp(
            headers={"Content-Type": "text/html"},
            raise_exc=err,
        ),
    )

    text = fetch_page.invoke({"url": "https://e.com", "snippet": "snip"})

    assert "Page fetch failed" in text
    assert "snip" in text


def test_fetch_page_unsupported_content_type_falls_back(monkeypatch):
    _patch_fetch(
        monkeypatch,
        _Resp(headers={"Content-Type": "application/pdf"}, body=b"%PDF"),
    )

    text = fetch_page.invoke({"url": "https://e.com", "snippet": "snip"})

    assert "unsupported content type" in text
    assert "snip" in text


def test_fetch_page_empty_body_falls_back(monkeypatch):
    _patch_fetch(
        monkeypatch,
        _Resp(headers={"Content-Type": "text/html"}, body=b"<html></html>"),
    )

    text = fetch_page.invoke({"url": "https://e.com", "snippet": "snip"})

    assert "empty page body" in text


def test_fetch_page_truncates_long_content(monkeypatch):
    body = ("<p>" + ("word " * 4000) + "</p>").encode()
    _patch_fetch(
        monkeypatch,
        _Resp(headers={"Content-Type": "text/html"}, body=body),
    )

    text = fetch_page.invoke({"url": "https://e.com", "snippet": ""})

    assert text.endswith("[Content truncated]")
    assert len(text) <= MAX_CHARS + len("\n\n[Content truncated]")


def test_fetch_page_follows_redirect(monkeypatch):
    _patch_fetch(
        monkeypatch,
        [
            _Resp(redirect=True, headers={"Location": "https://e.com/final"}),
            _Resp(headers={"Content-Type": "text/html"}, body=b"<p>Done</p>"),
        ],
    )

    text = fetch_page.invoke({"url": "https://e.com", "snippet": ""})

    assert "Done" in text


def test_fetch_page_too_many_redirects(monkeypatch):
    _patch_fetch(
        monkeypatch,
        _Resp(redirect=True, headers={"Location": "https://e.com/loop"}),
    )

    with pytest.raises(RuntimeError, match="Too many redirects"):
        fetch_page.invoke({"url": "https://e.com", "snippet": ""})


def test_fetch_page_request_exception_falls_back(monkeypatch):
    monkeypatch.setattr(
        "tools.web_search._resolve_and_validate", lambda url: "93.184.216.34"
    )

    def _boom(*_a, **_k):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr("tools.web_search.requests.get", _boom)

    text = fetch_page.invoke({"url": "https://e.com", "snippet": "snip"})

    assert "Page fetch failed" in text
    assert "snip" in text
