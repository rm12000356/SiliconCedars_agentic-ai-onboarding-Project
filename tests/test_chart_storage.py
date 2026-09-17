from __future__ import annotations

import asyncio

import pytest

from services.chart_storage import LocalChartStorage


def test_upload_read_delete_roundtrip(tmp_path):
    storage = LocalChartStorage(base_dir=tmp_path)
    key = "user-uuid/elem-uuid/chart.png"

    result = asyncio.run(storage.upload_file(key, b"PNGDATA", mime="image/png"))

    assert result["object_key"] == key
    assert result["url"].startswith("/charts/")

    url = asyncio.run(storage.get_read_url(key))
    assert url == result["url"]

    path = storage.resolve(key)
    assert path.read_bytes() == b"PNGDATA"

    assert asyncio.run(storage.delete_file(key)) is True
    assert not path.exists()


def test_resolve_stays_inside_base_dir(tmp_path):
    storage = LocalChartStorage(base_dir=tmp_path)

    # A leading slash must not turn into an absolute path escape.
    inside = storage.resolve("/etc/passwd")
    assert inside.is_relative_to(storage.base_dir)


def test_resolve_rejects_traversal(tmp_path):
    storage = LocalChartStorage(base_dir=tmp_path)

    with pytest.raises(ValueError):
        storage.resolve("../escape.txt")
    with pytest.raises(ValueError):
        storage.resolve("a/../../escape.txt")
    with pytest.raises(ValueError):
        storage.resolve("")


def test_token_is_opaque_and_reversible(tmp_path):
    storage = LocalChartStorage(base_dir=tmp_path)
    key = "user/elem/chart name.png"

    token = storage.encode_token(key)

    assert "/" not in token
    assert storage.decode_token(token) == key


def test_owner_of_returns_first_segment(tmp_path):
    storage = LocalChartStorage(base_dir=tmp_path)

    assert storage.owner_of("user-uuid/elem-uuid/chart.png") == "user-uuid"
    assert storage.owner_of("/user-uuid/elem-uuid/chart.png") == "user-uuid"
