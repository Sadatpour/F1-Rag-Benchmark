from __future__ import annotations

from urllib.parse import quote


JOLPICA_API_BASE = "https://api.jolpi.ca/ergast/f1"
JOLPICA_ATTRIBUTION = "Jolpica F1 API (Ergast-compatible)"
WIKIPEDIA_ARTICLE_BASE = "https://en.wikipedia.org/wiki/"


def jolpica_source_url(source: str) -> str:
    """Return a safe, directly browsable API URL for a Jolpica source ID."""
    if not source.startswith("Jolpica:"):
        return JOLPICA_API_BASE + "/"

    parts = source.removeprefix("Jolpica:").strip("/").split("/")
    if len(parts) < 2 or any(part in {"", ".", ".."} for part in parts):
        return JOLPICA_API_BASE + "/"

    if len(parts) == 3 and parts[2] == "schedule" and parts[0].isdigit():
        parts = parts[:2]
    elif len(parts) == 2 and parts[1] == "schedule" and parts[0].isdigit():
        parts = [parts[0]]
    path = "/".join(quote(part, safe="") for part in parts)
    return f"{JOLPICA_API_BASE}/{path}/?format=json"


def wikipedia_source_url(title: str) -> str:
    """Return a canonical English Wikipedia URL for a page title."""
    normalized_title = title.strip().replace(" ", "_")
    if not normalized_title:
        return WIKIPEDIA_ARTICLE_BASE
    return WIKIPEDIA_ARTICLE_BASE + quote(normalized_title, safe="()!,~*'-._")
