from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from chainlit.data.storage_clients.base import BaseStorageClient

DEFAULT_STORAGE_DIR = ".chainlit_charts"


class LocalChartStorage(BaseStorageClient):
    """Local-filesystem storage client. Without one, Chainlit's data layer
    never persists elements and chart images vanish on thread reload; files
    are written under ``base_dir`` and served via an opaque token URL."""
    def __init__(self, base_dir: str | os.PathLike[str] | None = None) -> None:
        self.base_dir = Path(
            base_dir or os.getenv("CHART_STORAGE_DIR", DEFAULT_STORAGE_DIR)
        ).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def encode_token(object_key: str) -> str:
        raw = base64.urlsafe_b64encode(object_key.encode("utf-8")).decode("ascii")
        return raw.rstrip("=")

    @staticmethod
    def decode_token(token: str) -> str:
        padded = token + "=" * (-len(token) % 4)
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")

    def resolve(self, object_key: str) -> Path:
        """Map an object key to a path inside base_dir, rejecting traversal."""
        key = object_key.strip().lstrip("/\\")
        if not key:
            raise ValueError("Empty object key")
        candidate = (self.base_dir / key).resolve()
        if not candidate.is_relative_to(self.base_dir):
            raise ValueError(f"Object key escapes storage dir: {object_key!r}")
        return candidate

    def url_for(self, object_key: str) -> str:
        return f"/charts/{self.encode_token(object_key)}"

    @staticmethod
    def owner_of(object_key: str) -> str:
        """First path segment of an object key (the owning user's id)."""
        return object_key.strip().lstrip("/\\").split("/", 1)[0]

    async def upload_file(
        self,
        object_key: str,
        data: bytes | str,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
        content_disposition: str | None = None,
    ) -> dict[str, Any]:
        path = self.resolve(object_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{object_key!r} already exists")
        if isinstance(data, str):
            data = data.encode("utf-8")
        path.write_bytes(data)
        return {"object_key": object_key, "url": self.url_for(object_key)}

    async def delete_file(self, object_key: str) -> bool:
        try:
            self.resolve(object_key).unlink()
            return True
        except FileNotFoundError:
            return False

    async def get_read_url(self, object_key: str) -> str:
        return self.url_for(object_key)

    async def close(self) -> None:
        return None
