from __future__ import annotations

import pytest

from tools.web_search import web_search


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
