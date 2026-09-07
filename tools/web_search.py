"""Web search tools (Tavily / DuckDuckGo)."""
from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from langchain.tools import tool
from ddgs import DDGS

MAX_CHARS = 4500          # ~ roughly 1000-1200 tokens
REQUEST_TIMEOUT = 12      # seconds
MAX_RESULTS = 5          

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
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=MAX_RESULTS))

            # Normalize the provider response into a clean, provider-agnostic shape
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in raw_results
            ]

    except Exception as e:
        # Never return error information as if it were a successful result
        raise RuntimeError(
            f"Web search failed: {type(e).__name__}: {e}"
        ) from e


@tool
def fetch_page(url: str) -> str:
    """
    Fetch and extract the main text content of a webpage.
    Returns a cleaned, truncated version of the page content
    suitable for research (max ~4500 characters).
    
    Returns an empty string if the page has no usable text.
    Raises an exception if the request itself fails.
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")

    # Basic validation
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http/https URLs are allowed. Got: {url}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"
        }
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        # Parse HTML and extract readable text
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if not clean_text:
            return ""   # successful fetch but no usable content

        # Truncate to hard limit
        if len(clean_text) > MAX_CHARS:
            clean_text = clean_text[:MAX_CHARS] + "\n\n[Content truncated]"

        return clean_text

    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to fetch page: {type(e).__name__}: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Error processing page content: {type(e).__name__}: {e}"
        ) from e