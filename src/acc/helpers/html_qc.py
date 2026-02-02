"""Light-weight heuristics to decide whether a fetched HTML page is useful.

Catches soft-failures like Cloudflare challenges, login interstitials,
empty stubs so we can retry with Selenium only when necessary.
"""
from __future__ import annotations

import re
from typing import Final

_BLOCK_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"cloudflare.+verify you are human", re.I),
    re.compile(r"checking your browser before accessing", re.I),
    re.compile(r"please enable javascript", re.I),
    re.compile(r"requires?\s+javascript", re.I),
    re.compile(r"javascript is disabled", re.I),
    re.compile(r"access denied", re.I),
    re.compile(r"error \d{3}", re.I),
    re.compile(r"<title>\s*login", re.I),
]

_MIN_READABLE_CHARS: Final[int] = 400
_MIN_HTML_BYTES: Final[int] = 1024
_MAX_SCRIPT_BYTE_RATIO: Final[float] = 0.30

_SCRIPT_RE = re.compile(
    r"<script\b[^>]*>.*?</script>",
    re.IGNORECASE | re.DOTALL,
)


def is_useful(html: str) -> bool:
    """Return True when html likely contains meaningful human content."""
    if len(html) < _MIN_HTML_BYTES:
        return False

    lowered = html.lower()
    if any(p.search(lowered) for p in _BLOCK_PATTERNS):
        return False

    if _script_byte_ratio(html) > _MAX_SCRIPT_BYTE_RATIO:
        return False

    try:
        import trafilatura
    except ImportError:
        return True

    extracted: str = trafilatura.extract(html, include_links=False, output_format="txt") or ""
    text_len = len(extracted)
    if text_len < _MIN_READABLE_CHARS or text_len / len(html) < 0.05:
        return False

    return True


def _script_byte_ratio(html: str) -> float:
    """Fraction of total HTML occupied by <script>...</script> blocks."""
    script_bytes = sum(len(match.group(0)) for match in _SCRIPT_RE.finditer(html))
    return script_bytes / max(len(html), 1)
