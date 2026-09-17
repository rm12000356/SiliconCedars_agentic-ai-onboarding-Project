import ipaddress
import socket
import threading
from urllib.parse import urlparse, urljoin

import requests
import urllib3.util.connection as urllib3_conn
from bs4 import BeautifulSoup
from langchain.tools import tool
from ddgs import DDGS

MAX_CHARS = 4500          # ~ roughly 1000-1200 tokens
REQUEST_TIMEOUT = 12      # seconds
MAX_RESULTS = 5
MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 5_000_000  # 5 MB hard cap before we give up parsing
ALLOWED_CONTENT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")
ALLOWED_PORTS = (80, 443)

_original_create_connection = urllib3_conn.create_connection
_PIN_LOCK = threading.Lock()

def _is_blocked_ip(ip_str: str) -> bool:
    """
    True if this IP should never be fetched: loopback, private ranges,
    link-local (includes the cloud metadata endpoint 169.254.169.254),
    reserved, or multicast.
    """
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )




def _resolve_and_validate(url: str) -> str:
    """
    Validates scheme, port, hostname resolution, and every resolved IP.
    Returns the first safe resolved IP, which the caller then pins the
    actual TCP connection to, closing the DNS-rebinding gap: without
    pinning, a malicious DNS server could return a safe IP for this
    check and a blocked IP (e.g. 127.0.0.1) for the real request.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http/https URLs are allowed. Got: {url}")
    if not parsed.hostname:
        raise ValueError(f"URL has no hostname: {url}")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise ValueError(f"Port {port} is not allowed for URL {url!r}, only 80/443.")

    try:
        resolved = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve hostname {parsed.hostname!r}: {e}")

    safe_ip = None
    for _, _, _, _, sockaddr in resolved:
        ip_str = str(sockaddr[0])
        if _is_blocked_ip(ip_str):
            raise ValueError(
                f"URL {url!r} resolves to a blocked address ({ip_str}), "
                "internal/private/link-local targets are not allowed."
            )
        safe_ip = safe_ip or ip_str

    if safe_ip is None:
        raise ValueError(f"No usable resolved address for {url!r}")

    return safe_ip


def _pin_connection(pinned_ip: str):

    def _pinned(address, *args, **kwargs):
        host, port = address
        return _original_create_connection((pinned_ip, port), *args, **kwargs)
    return _pinned


@tool
def web_search(query: str) -> list[dict]:
    """
    Search the public web for information.
    Returns a list of results, each containing:
    - title
    - url
    - snippet

    Returns an empty list if no results are found.
    Raises an exception if the search itself fails.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")

    try:
        with DDGS(timeout=REQUEST_TIMEOUT) as ddgs:
            raw_results = list(ddgs.text(query, max_results=MAX_RESULTS))

            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in raw_results
            ]

    except Exception as e:
        raise RuntimeError(
            f"Web search failed: {type(e).__name__}: {e}"
        ) from e


def _snippet_fallback(url: str, snippet: str, reason: str) -> str:
    text = (snippet or "").strip()
    if not text:
        raise RuntimeError(
            f"Failed to fetch page ({reason}) and no search snippet was provided. url={url}"
        )
    return (
        f"[Page fetch failed: {reason}]\n"
        f"Source: {url}\n"
        f"Using DuckDuckGo snippet as fallback:\n{text}"
    )


@tool
def fetch_page(url: str, snippet: str = "") -> str:
    """
    Fetch and extract the main text of a webpage (max ~4500 characters).

    Always pass `snippet` from the matching web_search result. If the
    page cannot be fetched (403, timeout, non-HTML, empty body), that
    snippet is returned as a labeled fallback instead of failing the
    research pass. Internal/private URLs still raise (no snippet bypass).
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")

    current_url = url
    response = None

    try:
        for _ in range(MAX_REDIRECTS + 1):
            pinned_ip = _resolve_and_validate(current_url)

            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"
            }

            with _PIN_LOCK:
                previous_create_connection = urllib3_conn.create_connection
                urllib3_conn.create_connection = _pin_connection(pinned_ip)
                try:
                    response = requests.get(
                        current_url,
                        headers=headers,
                        timeout=REQUEST_TIMEOUT,
                        allow_redirects=False,
                        stream=True,
                    )
                finally:
                    urllib3_conn.create_connection = previous_create_connection

            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                response.close()
                response = None
                if not location:
                    raise RuntimeError("Redirect response missing Location header")
                current_url = urljoin(current_url, location)
                continue
            break
        else:
            raise RuntimeError(f"Too many redirects (> {MAX_REDIRECTS})")

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", "?")
            return _snippet_fallback(url, snippet, f"HTTP {code}")

        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if not content_type or not any(content_type.startswith(t) for t in ALLOWED_CONTENT_TYPES):
            return _snippet_fallback(
                url, snippet, f"unsupported content type {content_type!r}"
            )

        raw_bytes = b""
        for chunk in response.iter_content(chunk_size=65536):
            raw_bytes += chunk
            if len(raw_bytes) > MAX_RESPONSE_BYTES:
                break
        text_content = raw_bytes.decode(response.encoding or "utf-8", errors="replace")

        soup = BeautifulSoup(text_content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if not clean_text:
            return _snippet_fallback(url, snippet, "empty page body")

        if len(clean_text) > MAX_CHARS:
            clean_text = clean_text[:MAX_CHARS] + "\n\n[Content truncated]"

        return clean_text

    except requests.RequestException as e:
        try:
            return _snippet_fallback(url, snippet, f"{type(e).__name__}: {e}")
        except RuntimeError:
            raise RuntimeError(
                f"Failed to fetch page: {type(e).__name__}: {e}"
            ) from e
    except ValueError:
        # SSRF / scheme / port / blocked IP — never fall back to snippet
        raise
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            f"Error processing page content: {type(e).__name__}: {e}"
        ) from e
    finally:
        if response is not None:
            response.close()