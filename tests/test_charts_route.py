from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires_db

pytestmark = [pytest.mark.integration, requires_db]


def test_charts_route_serves_image_and_is_not_shadowed(tmp_path, monkeypatch):
    """
    Regression: Chainlit's SPA catch-all ("/{full_path:path}") used to swallow
    /charts/{token}, returning index.html. The route is now spliced ahead of the
    included router, so it must serve the PNG.
    """
    import chat
    from chainlit.auth import get_current_user
    from chainlit.server import app

    monkeypatch.setattr(chat.CHART_STORAGE, "base_dir", tmp_path.resolve())

    key = "user-uuid/elem-uuid/regression-chart.png"
    asyncio.run(
        chat.CHART_STORAGE.upload_file(key, b"\x89PNG\r\n\x1a\n", mime="image/png")
    )
    token = chat.CHART_STORAGE.encode_token(key)

    app.dependency_overrides[get_current_user] = lambda: object()
    try:
        response = TestClient(app).get(f"/charts/{token}")
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/png")
    assert "text/html" not in response.headers["content-type"]
