"""Content fetching service for the automated compliance check system.

This module provides functionality to fetch web content, including HTML pages,
robots.txt, and discovered Terms & Conditions pages.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import mimetypes
import os
import re
import time
from asyncio import Semaphore
from typing import Dict, List, Tuple
from urllib.parse import unquote, urljoin, urlparse, urlunparse

import filetype
import httpx
import tldextract
import trafilatura
from async_lru import alru_cache
from bs4 import BeautifulSoup
from lxml import etree as LET
from pydantic import HttpUrl
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.acc.config import settings
from src.acc.helpers.html_qc import is_useful
from src.acc.helpers.validators import validate_url
from src.acc.infra.http_client import (
    get_http_client,
    get_selenium_driver,
    get_statuscode_from_selenium,
)
from src.acc.infra.storage import storage_service
from src.acc.models import (
    DiscoveredArtifacts,
    DiscoveryMethod,
    FetchedContent,
    RelatedLegalDocument,
    TermsVariant,
)

logger = logging.getLogger(__name__)

_semaphores: dict = {}

def get_fetch_semaphore() -> asyncio.Semaphore:
    """Get or create a semaphore for the current event loop."""
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        return asyncio.Semaphore(settings.max_connections)

    if loop_id not in _semaphores:
        _semaphores[loop_id] = asyncio.Semaphore(settings.max_connections)

    return _semaphores[loop_id]

GLOBAL_FETCH_SEM = None  # Will be replaced by get_fetch_semaphore() calls

__all__ = [
    'fetch_url',
    'fetch_primary_html',
    'fetch_robots_txt',
    'discover_terms_artifacts',
]


class DomainRateLimiter:
    """
    Rate limiter for domain-specific request management.

    Attributes:
        delay: Delay between requests to the same domain.
        domain_semaphores: Dictionary of semaphores for each domain per event loop.
        last_request_time: Dictionary of last request time for each domain.
    """

    def __init__(self, requests_per_second: float = settings.requests_per_second):
        """Initialize the rate limiter with configured RPS.

        Args:
            requests_per_second: Maximum number of requests per second to the same domain.
        """
        self.delay = 1.0 / requests_per_second
        self.domain_semaphores: Dict[tuple, Semaphore] = {}
        self.last_request_time: Dict[str, float] = {}

    def get_semaphore(self, domain: str) -> Semaphore:
        """Get or create a semaphore for a specific domain in the current event loop.

        Args:
            domain: The domain to get a semaphore for.

        Returns:
            A semaphore for the specified domain.
        """
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            return Semaphore(1)

        key = (loop_id, domain)
        if key not in self.domain_semaphores:
            self.domain_semaphores[key] = Semaphore(1)
            if domain not in self.last_request_time:
                self.last_request_time[domain] = 0
        return self.domain_semaphores[key]

    async def acquire(self, domain: str):
        """Acquire the semaphore for a domain and enforce rate limiting.

        Args:
            domain: The domain to acquire the semaphore for.
        """
        semaphore = self.get_semaphore(domain)
        await semaphore.acquire()

        current_time = asyncio.get_event_loop().time()
        last_time = self.last_request_time.get(domain, 0)
        time_since_last_request = current_time - last_time

        if time_since_last_request < self.delay:
            wait_time = self.delay - time_since_last_request
            logger.info(f'Rate limiting for {domain}: waiting {wait_time:.2f}s')
            await asyncio.sleep(wait_time)

        self.last_request_time[domain] = asyncio.get_event_loop().time()

    def release(self, domain: str):
        """Release the semaphore for a domain.

        Args:
            domain: The domain to release the semaphore for.
        """
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
            key = (loop_id, domain)
            if key in self.domain_semaphores:
                self.domain_semaphores[key].release()
        except RuntimeError:
            pass


_rate_limiters: dict = {}

def get_rate_limiter() -> DomainRateLimiter:
    """Get or create a rate limiter for the current event loop."""
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        return DomainRateLimiter()

    if loop_id not in _rate_limiters:
        _rate_limiters[loop_id] = DomainRateLimiter()

    return _rate_limiters[loop_id]

rate_limiter = None

SITEMAP_NS = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

_TEXTUAL_APPLICATION = [
    'application/json',
    'application/javascript',
    'application/xml',
    'application/x-www-form-urlencoded',
    "image/svg+xml",
    "text/"
]

# explicit path patterns - exact matches only
# EXPLICIT_PATTERNS = [
#     re.compile(r'/terms/?$'),
#     re.compile(r'/terms-of-use/?$'),
#     re.compile(r'/terms-of-service/?$'),
#     re.compile(r'/terms-and-conditions/?$'),
#     re.compile(r'/termsofuse/?$'),
#     re.compile(r'/termsofservice/?$'),
#     re.compile(r'/termsandconditions/?$'),
#     re.compile(r'/legal/terms/?$'),
#     re.compile(r'/legal/terms-of-use/?$'),
#     re.compile(r'/legal/terms-of-service/?$'),
#     re.compile(r'/legal/terms-and-conditions/?$'),
#     re.compile(r'/tos/?$'),
#     re.compile(r'/agb/?$'),
#     re.compile(r'/aab/?$'),
#     re.compile(r'/allgemeine-auftragsbedingungen/?$'),
# ]
EXPLICIT_PATTERNS = [
    re.compile(r'/terms/?$'),
    re.compile(r'/terms[-_]of[-_]use(?:[-\w/]*)/?$'),
    re.compile(r'/terms[-_]of[-_]service(?:[-\w/]*)/?$'),
    re.compile(r'/terms[-_]and[-_]conditions(?:[-\w/]*)/?$'),
    re.compile(r'/termsofuse/?$'),
    re.compile(r'/termsofservice/?$'),
    re.compile(r'/termsandconditions/?$'),
    re.compile(r'/legal/terms/?$'),
    re.compile(r'/legal/terms[-_]of[-_]use(?:[-\w/]*)/?$'),
    re.compile(r'/legal/terms[-_]of[-_]service(?:[-\w/]*)/?$'),
    re.compile(r'/legal/terms[-_]and[-_]conditions(?:[-\w/]*)/?$'),
    re.compile(r'/tos/?$'),
    re.compile(r'/agb/?$'),
    re.compile(r'/aab/?$'),
    re.compile(r'/allgemeine-auftragsbedingungen/?$'),
]
# Add SITEMAP_PRIORITY_KEYWORDS near other constants or imports
SITEMAP_PRIORITY_KEYWORDS = [
    "home", "about", "legal", "terms", "privacy", "contact", "page", "disclaimer",
    "security", "agreement", "policy", "conditions", "use", "company", "service",
    "main"
]

def _is_explicit(url: str) -> bool:
    """Check if URL matches explicit T&C patterns.

    Args:
        url: The URL to check.

    Returns:
        bool: True if the URL matches an explicit T&C pattern, False otherwise.
    """
    return any(p.search(url) for p in EXPLICIT_PATTERNS)


def _is_binary_content(content: bytes, headers: dict) -> bool:
    """
    Detect whether the fetched content is binary.
    More tolerant for missing/unknown Content-Type headers so that HTML
    without headers is not misclassified as binary.
    """

    def _get_header(name: str):
        for k, v in headers.items():
            if k.lower() == name.lower():
                return v
        return None

    # 1- first Check Content-Type header if present
    ctype = _get_header("Content-Type")
    if ctype:
        ctype = ctype.split(";")[0].strip().lower()
        if any(ctype.startswith(t) for t in _TEXTUAL_APPLICATION):
            return False
        # Known binary types
        if any(ctype.startswith(t) for t in (
            "image/", "audio/", "video/", "application/pdf",
            "application/zip", "application/x-gzip", "application/vnd.ms-",
            "application/vnd.openxmlformats-"
        )):
            return True
        # Unknown application/*
        if ctype.startswith("application/"):
            # Fall through to sniffing instead of assuming binary
            pass


    # 2- second, Check file extension
    # Only applies if URL was available in headers
    url_hint = _get_header("Content-Location") or _get_header("X-Request-URL")
    if url_hint:
        guessed_type, _ = mimetypes.guess_type(url_hint)
        if guessed_type:
            if guessed_type.startswith("text/") or guessed_type in _TEXTUAL_APPLICATION:
                return False
            return True

    # 3- third, Sniff first bytes
    # Look for binary numbers
    magic_bytes = content[:16]
    binary_signatures = [
        b"\x89PNG",  # PNG
        b"\xFF\xD8\xFF",  # JPEG
        b"%PDF",  # PDF
        b"PK\x03\x04",  # ZIP/docx/xlsx/pptx
        b"GIF87a", b"GIF89a",  # GIF
    ]
    for sig in binary_signatures:
        if magic_bytes.startswith(sig):
            return True

    # 4- fourth, Try to decode and look for HTML markers
    try:
        sample_text = content[:1024].decode("utf-8", errors="ignore").lower()
        if "<html" in sample_text or "<!doctype html" in sample_text:
            return False
    except Exception:
        pass

    # 5- fifth, Try a quick UTF-16 decode if UTF-8 failed
    try:
        sample_text = content[:1024].decode("utf-16", errors="ignore").lower()
        if "<html" in sample_text or "<!doctype html" in sample_text:
            return False
    except Exception:
        pass

    # 6) Default: assume text if we got here and decode didn't crash badly
    try:
        content[:512].decode("utf-8")
        return False
    except Exception:
        return True

    # mime = (content_type or '').split(';', 1)[0].strip().lower()

    # if mime.startswith('text/') or mime in _TEXTUAL_APPLICATION:
    #     return False
    # if mime == 'application/octet-stream':
    #     return True  # unknown binary by convention
    # if mime and not mime.startswith(('text/', 'application/')):
    #     # image/*, audio/*, video/*, font/*, model/* … are all binary families
    #     return True

    # filename = unquote(urlparse(url).path.rsplit('/', 1)[-1])
    # guess, _ = mimetypes.guess_type(filename, strict=False)
    # if guess:
    #     if guess.startswith('text/') or guess in _TEXTUAL_APPLICATION:
    #         return False
    #     return True

    # if sniff:
    #     kind = filetype.guess(sniff)
    #     if kind:
    #         return not kind.mime.startswith('text/')

    # return True


def extract_text_trafilatura(html_content: str) -> Tuple[str, List[Dict[str, str]]]:
    """Extract main text content and links from HTML using trafilatura.

    This uses the trafilatura library which is specifically designed for extracting
    clean text from web pages while removing boilerplate, navigation, etc. Also extracts
    markdown-style links from the content for further processing of related legal documents.

    Args:
        html_content: The raw HTML string to extract text from.

    Returns:
        A tuple containing:
            - A string with the extracted text content (empty string if extraction fails)
            - A list of dictionaries with link information {link_text, url}
    """
    try:
        json_result = trafilatura.extract(
            html_content, include_links=True, output_format='json'
        )

        if not json_result:
            logger.warning('trafilatura extraction returned None')
            return '', []

        data = json.loads(json_result)
        extracted_text = data.get('text', '')

        links = []
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(link_pattern, extracted_text):
            link_text = match.group(1)
            link_url = match.group(2)
            links.append({'link_text': link_text, 'url': link_url})

        # Note: We keep the links in the text since they provide context
        # This helps compliance manager understand the references

        return extracted_text, links
    except Exception as e:
        logger.error(f'Error extracting text with trafilatura: {e}')
        return '', []


def _wait_for_full_render(driver, timeout: int = 15) -> None:
    """Block until `document.readyState == 'complete'` *and* no XHR/fetch."""
    def page_is_idle(_driver) -> bool:
        if _driver.execute_script("return document.readyState") != "complete":
            return False
        return bool(
            _driver.execute_script(
                "return (window.jQuery ? jQuery.active === 0 : true) "
                "&& window.pendingFetches === 0;"
            )
        )

    WebDriverWait(driver, timeout, poll_frequency=0.4).until(page_is_idle)


def _fetch_with_selenium(url: str) -> FetchedContent:
    """
    Fetch the URL using Selenium and return the rendered HTML.
    """

    driver = None
    try:
        driver = get_selenium_driver()
        driver.get(url)

        # Ensure tracker exists if CDP pre-injection failed/unavailable
        if getattr(driver, "_post_nav_inject", True):
            driver.execute_script("""
                if (window.pendingFetches === undefined) {
                    window.pendingFetches = 0;
                    const _origFetch = window.fetch;
                    window.fetch = function () {
                        window.pendingFetches++;
                        return _origFetch.apply(this, arguments)
                            .finally(() => window.pendingFetches--);
                    };
                }
            """)
        # Two-phase wait:
        # 1) Wait for DOM readyState 'complete' up to fetch_timeout.
        # 2) Best-effort short (3s) network-idle wait; do not fail on timeout.
        try:
            WebDriverWait(driver, settings.fetch_timeout, poll_frequency=0.4).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            try:
                WebDriverWait(driver, 3, poll_frequency=0.4).until(
                    lambda d: d.execute_script(
                        "const pf=window.pendingFetches;"
                        "return (window.jQuery ? jQuery.active === 0 : true) && (!pf || pf===0);"
                    )
                )
            except TimeoutException:
                try:
                    logger.info(f"Network idle wait skipped after 3s for {url}; returning DOM")
                except NameError:
                    pass
        except TimeoutException:
            try:
                logger.warning(
                    "ReadyState wait timed out after %.1fs for %s; returning current DOM",
                    settings.fetch_timeout, url
                )
            except NameError:
                pass

        page_source: str = driver.page_source or ""
        status_code = get_statuscode_from_selenium(driver, url)

        return FetchedContent(
            request_url=url,
            final_url=driver.current_url,
            status_code=status_code,
            content_type="text/html",
            content=page_source,
            is_binary=False,
        )
    except Exception as exc:
        err = f"Selenium error for {url}: {type(exc).__name__}: {exc}"
        try:
            logger.warning(err)
        except NameError:
            pass
        return FetchedContent(request_url=url, error=err, content_type=None)

    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:  # pylint: disable=broad-except
                pass


async def _fetch_text_http_only(url: str, detect_binary: bool = False) -> FetchedContent:
    """HTTP-only fetch with optional binary detection and gzip decompression.

    - Never falls back to Selenium
    - When detect_binary=True, detects binary payloads and stores them via `storage_service`
    - Returns text content when possible; sets error on HTTP >= 400
    """
    domain = urlparse(url).netloc
    await get_rate_limiter().acquire(domain)
    request_url_str = str(url)
    try:
        logger.info('HTTP-only fetch: %s', request_url_str)
        http_client = get_http_client()
        response = await http_client.get(request_url_str)

        content_type = response.headers.get('content-type', '')
        final_url = str(response.url)
        status_code = response.status_code
        etag = response.headers.get('etag')
        last_modified = response.headers.get('last-modified')

        raw = response.content
        ct_lower = (content_type or '').lower()
        is_gz = (
            request_url_str.lower().endswith('.gz')
            or 'application/gzip' in ct_lower
            or 'application/x-gzip' in ct_lower
        )
        if is_gz:
            try:
                raw = gzip.decompress(raw)
                if 'xml' not in ct_lower:
                    content_type = 'application/xml'
            except Exception as e:
                err = f'Failed to decompress gzip for {request_url_str}: {e}'
                logger.warning(err)
                return FetchedContent(
                    request_url=request_url_str,
                    final_url=final_url,
                    status_code=status_code,
                    content_type=content_type,
                    content=None,
                    is_binary=True,
                    error=err,
                    etag=etag,
                    last_modified=last_modified,
                )

        # If the response is OK and detect_binary=True, check for binary payloads and store them
        try:
            if detect_binary and status_code < 400:
                if _is_binary_content(raw, dict(response.headers)):
                    # Derive or refine content-type
                    eff_ct = (content_type or '').split(';', 1)[0].strip() or None
                    if not eff_ct:
                        try:
                            kind = filetype.guess(raw)
                            if kind and getattr(kind, 'mime', None):
                                eff_ct = kind.mime
                        except Exception:
                            eff_ct = None
                    if not eff_ct:
                        eff_ct = 'application/octet-stream'

                    try:
                        storage_uri, url_hash = await storage_service.store_binary_content(
                            content=raw,
                            source_url=final_url or request_url_str,
                            content_type=eff_ct,
                            purpose='legal',
                        )
                    except Exception as store_exc:  # pragma: no cover - storage failure path
                        err = f"Failed to store binary content for {request_url_str}: {store_exc}"
                        logger.error(err)
                        return FetchedContent(
                            request_url=request_url_str,
                            final_url=final_url,
                            status_code=status_code,
                            content_type=eff_ct,
                            content=None,
                            is_binary=True,
                            error=err,
                            etag=etag,
                            last_modified=last_modified,
                        )

                    return FetchedContent(
                        request_url=request_url_str,
                        final_url=final_url,
                        status_code=status_code,
                        content_type=eff_ct,
                        content=None,
                        is_binary=True,
                        binary_content_size=len(raw),
                        storage_uri=storage_uri,
                        etag=etag,
                        last_modified=last_modified,
                        url_hash=url_hash,
                    )
        except Exception as detect_exc:  # pragma: no cover - conservative guard
            logger.warning(
                'Binary detection failed for %s: %s', request_url_str, detect_exc
            )

        try:
            text = raw.decode(response.encoding or 'utf-8', errors='replace')
        except Exception:
            text = raw.decode('utf-8', errors='replace')

        # Mark non-2xx as error to keep is_success false
        if status_code >= 400:
            return FetchedContent(
                request_url=request_url_str,
                final_url=final_url,
                status_code=status_code,
                content_type=content_type,
                content=text,
                is_binary=False,
                etag=etag,
                last_modified=last_modified,
                error=f'HTTP {status_code}',
            )

        return FetchedContent(
            request_url=request_url_str,
            final_url=final_url,
            status_code=status_code,
            content_type=content_type,
            content=text,
            is_binary=False,
            etag=etag,
            last_modified=last_modified,
        )
    except (httpx.HTTPStatusError, httpx.RequestError) as http_err:
        logger.warning('HTTP-only fetch failed: %s', http_err)
        return FetchedContent(request_url=request_url_str, error=str(http_err), content_type=None)
    except Exception as exc:  # pylint: disable=broad-except
        err = f"Unexpected error fetching {request_url_str}: {type(exc).__name__}: {exc}"
        logger.error(err, exc_info=True)
        return FetchedContent(request_url=request_url_str, error=err, content_type=None)
    finally:
        get_rate_limiter().release(domain)


async def _fetch_html_with_fallback(url: str, detect_binary: bool = False) -> FetchedContent:
    """Fetch HTML via HTTP first, then fallback to Selenium if content unusable.

    When detect_binary=True, binary content is detected at the HTTP step and returned
    immediately (no Selenium fallback).
    """
    http_res = await _fetch_text_http_only(url, detect_binary=detect_binary)
    # If HTTP-only fetch succeeded with binary content, return it (no Selenium fallback)
    if detect_binary and http_res.is_success and getattr(http_res, "is_binary", False):
        return http_res
    if http_res.is_success and http_res.content and is_useful(http_res.content):
        return http_res
    logger.info('Falling back to Selenium for %s', url)
    return _fetch_with_selenium(url)


async def fetch_primary_html(base_url: HttpUrl) -> FetchedContent:
    """Fetches the primary HTML content from the root of the base URL's domain.

    Constructs the landing page URL (e.g., 'https://example.com/') from the
    base URL and fetches it using the common _fetch_url function.

    Args:
        base_url: The validated Pydantic HttpUrl of the source.

    Returns:
        A FetchedContent object for the HTML fetch attempt.
    """
    # Construct the landing page URL explicitly at the root of the domain
    # urlunparse components: (scheme, netloc, path, params, query, fragment)

    # landing_page_url_str = urlunparse((base_url.scheme, base_url.host, '/', '', '', ''))
    # logger.info(f'Constructed landing page URL for HTML fetch: {landing_page_url_str}')
    # return await _fetch_html_with_fallback(landing_page_url_str)
    return await _fetch_html_with_fallback(str(base_url))


async def fetch_robots_txt(base_url: HttpUrl) -> FetchedContent:
    """Fetches the robots.txt file from the root of the base URL's domain.

    Constructs the robots.txt URL (e.g., 'https://example.com/robots.txt')
    from the base URL and fetches it using the common _fetch_url function.

    Args:
        base_url: The validated Pydantic HttpUrl of the source.

    Returns:
        A FetchedContent object for the robots.txt fetch attempt.
    """
    # Build robots.txt on the registrable domain so that sub-domains such as
    # "info." are stripped (e.g., info.randstad.se -> randstad.se)
    extracted = tldextract.extract(base_url.host)
    # If extraction fails to find a suffix (rare), fall back to original host
    root_domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else base_url.host
    host_with_www = root_domain if root_domain.startswith('www.') else f'www.{root_domain}'
    robots_url_str = f"{base_url.scheme}://{host_with_www}/robots.txt"
    logger.info(f'Constructed robots.txt URL: {robots_url_str}')
    return await _fetch_text_http_only(robots_url_str)


async def fetch_url(url: str) -> FetchedContent:
    """Fetches content from a specific discovered URL.

    Uses the HTML-first strategy with Selenium fallback and binary detection.

    Args:
        url: The URL string to fetch.

    Returns:
        A FetchedContent object for the fetch attempt.
    """
    return await _fetch_html_with_fallback(url, detect_binary=True)


def extract_sitemap_urls_from_robots(robots_content: FetchedContent) -> List[str]:
    """Extracts sitemap URLs from robots.txt content.

    Args:
        robots_content: The fetched robots.txt content.

    Returns:
        A list of sitemap URLs found in robots.txt.
    """
    sitemap_pattern = re.compile(r'(?i)^\s*Sitemap\s*:\s*(.+)\s*$', re.MULTILINE)
    return [m.strip() for m in sitemap_pattern.findall(robots_content.content)]


def _tc_from_urlset(root: LET._Element) -> List[str]:
    """
    Extracts all explicit matches (e.g. /terms-of-use) and fallbacks.

    Args:
        root: The root element of the sitemap XML.

    Returns:
        A list of URLs found in the sitemap (limited by depth and max_urls).
    """
    locs = (
        loc.strip() for loc in root.xpath('//sm:url/sm:loc/text()', namespaces=SITEMAP_NS)
    )
    return [loc for loc in locs if loc and _is_explicit(loc)]


@alru_cache(maxsize=20_000)
async def parse_sitemap(sitemap_url: str) -> List[str]:
    """Parses a sitemap XML to extract all URLs with controlled recursion depth.

    Uses optimized XML processing with lxml and implements caching
    to avoid re-processing the same sitemap multiple times. Handles
    both standard and non-standard sitemap formats.

    Args:
        sitemap_url: The URL of the sitemap to parse.

    Returns:
        A list of URLs found in the sitemap (limited by depth and max_urls).
    """
    logger.info(f'Parsing sitemap: {sitemap_url}')
    # Use HTTP-only fetch to avoid Selenium and binary storage side effects
    sitemap_content = await _fetch_text_http_only(sitemap_url)

    if not sitemap_content.is_success or not sitemap_content.content:
        logger.warning(f'Failed to fetch sitemap at {sitemap_url}')
        return []

    try:
        root = LET.fromstring(sitemap_content.content.encode())
        logger.debug(f'Root tag for {sitemap_url}: {root.tag}')
    except Exception as e:
        logger.error(f'XML parsing error for sitemap {sitemap_url}: {e}')
        return []

    root_name = LET.QName(root).localname
    tc_urls: List[str] = []

    if root_name == 'sitemapindex':
        logger.info(f'Found sitemap index at {sitemap_url}')

        locs = root.xpath('//sm:sitemap/sm:loc/text()', namespaces=SITEMAP_NS)

        # Identify geo/language specific sitemaps only when the language code
        # appears **immediately after the domain**, e.g.:
        #   https://example.com/en/...
        #   https://example.com/en-us/...
        # but NOT https://example.com/sitemaps/jobs/en/...
        # We consider two cases:
        #   1. First path segment is a 2-letter code or code pair (en, en-us)
        #   2. Language code embedded directly in the filename at domain root

        geo_specific_sitemaps: list[str] = []
        regular_sitemaps: list[str] = []

        lang_segment_regex = re.compile(r'^[a-z]{2}([-_][a-z]{2})?$')  # en | en-us | en_us
        filename_lang_regex = re.compile(r'^[a-z]{2}([-_][a-z]{2})?\.')  # en. | en-us.

        for loc in locs:
            parsed = urlparse(loc.lower())
            path = parsed.path.lstrip('/')  # remove leading /

            if not path:
                regular_sitemaps.append(loc)
                continue

            first_segment = path.split('/', 1)[0]
            filename = first_segment if '/' not in path else path.split('/')[-1]

            is_geo_first_segment = bool(lang_segment_regex.fullmatch(first_segment))
            is_geo_filename = bool(filename_lang_regex.match(filename))

            if is_geo_first_segment or is_geo_filename:
                geo_specific_sitemaps.append(loc)
            else:
                regular_sitemaps.append(loc)

        logger.info(
            f"Found {len(geo_specific_sitemaps)} geo-specific and {len(regular_sitemaps)} regular sitemaps"
        )

        # Process all sitemaps (both geo-specific and regular) for comprehensive coverage
        all_candidate_sitemaps = geo_specific_sitemaps + regular_sitemaps

        # Prioritize child sitemaps based on keywords from all candidates
        prioritized_child_sitemap_urls = [
            loc for loc in all_candidate_sitemaps
            if any(keyword in loc.lower() for keyword in SITEMAP_PRIORITY_KEYWORDS)
        ]

        # If no keyword matches found, but we have geo-specific sitemaps, include them
        if not prioritized_child_sitemap_urls and geo_specific_sitemaps:
            logger.info(
                f"No keyword matches found, but including {len(geo_specific_sitemaps)} geo-specific sitemaps"
            )
            prioritized_child_sitemap_urls = geo_specific_sitemaps

        if not prioritized_child_sitemap_urls and all_candidate_sitemaps:
            logger.info(
                f"Sitemap index {sitemap_url} has {len(all_candidate_sitemaps)} child sitemaps, "
                f"but none matched priority keywords: {SITEMAP_PRIORITY_KEYWORDS}. "
                f"No child sitemaps from this index will be parsed."
            )
        elif prioritized_child_sitemap_urls:
            logger.info(
                f"Selected {len(prioritized_child_sitemap_urls)} child sitemaps from {sitemap_url} "
                f"(including {len([s for s in prioritized_child_sitemap_urls if s in geo_specific_sitemaps])} geo-specific)"
            )

        if prioritized_child_sitemap_urls:
            async def fetch_child(u: str):
                async with get_fetch_semaphore():
                    # Use HTTP-only fetch for child sitemaps as well
                    return await _fetch_text_http_only(u)

            child_responses = await asyncio.gather(*(fetch_child(u) for u in prioritized_child_sitemap_urls))

            for loc, child in zip(prioritized_child_sitemap_urls, child_responses):
                if not (child.is_success and child.content):
                    logger.warning(f'Failed to fetch child sitemap at {child.request_url}')
                    continue
                try:
                    child_root = LET.fromstring(child.content.encode())
                    child_root_name = LET.QName(child_root).localname

                    if child_root_name == 'urlset':
                        tc_urls.extend(_tc_from_urlset(child_root))
                    elif child_root_name == 'sitemapindex':
                        # Handle nested sitemap index (common for geo-specific sitemaps)
                        logger.info(f"Found nested sitemap index at {loc}, recursively parsing")
                        nested_tc_urls = await parse_sitemap(loc)
                        tc_urls.extend(nested_tc_urls)
                    else:
                        logger.warning(
                            f"Child sitemap {loc} from index {sitemap_url} is neither 'urlset' nor 'sitemapindex' "
                            f"(actual: {child_root_name}). Skipping."
                        )
                except Exception as e:
                    logger.error('XML parsing error in child sitemap %s: %s', loc, e)

    elif root_name == 'urlset':
        tc_urls = _tc_from_urlset(root)

    else:
        logger.warning(f'Unknown sitemap type: {root_name}')
        return []

    logger.info(f'Extracted {len(tc_urls)} URLs from sitemap {sitemap_url}')

    return tc_urls


def is_terms_url(url: str) -> bool:
    """Determines if a URL is likely a Terms & Conditions page.

    This function uses a general pattern-matching approach to identify URLs
    that likely point to Terms & Conditions pages, supporting a wide variety
    of naming conventions and URL structures across thousands of different sources.

    Args:
        url: The URL to check.

    Returns:
        True if the URL is likely a T&C page, False otherwise.
    """
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()

    filename = os.path.basename(path)

    path_segments = [segment.lower() for segment in path.split('/') if segment]

    terms_keywords = ['terms', 'tos', 'eula', 'agreement', 'conditions', 'agb']

    # Check for exact path segment matches (like /terms/ or /tos/)
    for segment in path_segments:
        if segment in terms_keywords:
            return True

    # Check for compound terms (terms-of-service, terms-of-use, terms-and-conditions)
    compound_patterns = [
        r'terms[-_]of[-_]service',
        r'terms[-_]of[-_]use',
        r'terms[-_]and[-_]conditions',
    ]

    for pattern in compound_patterns:
        for segment in path_segments:
            if re.search(pattern, segment):
                return True

    # Broader check: see if 'terms' appears as part of any path segment
    # This will catch URLs like /legal/terms-before-2019-11-20/ or /terms-policy/
    for keyword in terms_keywords:
        if any(keyword in segment for segment in path_segments):
            return True

    # Also check if the terms keyword appears as a prefix in any path segment
    # This helps with URLs where 'terms' is part of a longer segment name
    for keyword in terms_keywords:
        if any(segment.startswith(keyword) for segment in path_segments):
            return True

    # Check filename for terms keywords (for document URLs like .pdf)
    # This supports cases like Cisco_General_Terms.pdf without special-casing
    if filename and any(keyword in filename for keyword in terms_keywords):
        return True

    return False


def extract_terms_urls_from_html(
    html_content: FetchedContent, base_url: str
) -> List[str]:
    """Extracts potential T&C URLs from HTML content.

    Args:
        html_content: The fetched HTML content.
        base_url: The base URL for resolving relative links.

    Returns:
        A list of potential T&C URLs found in the HTML.
    """
    terms_urls = []
    try:
        soup = BeautifulSoup(html_content.content, 'html.parser')

        # TODOs:
        # - find keywords for different languages and add them to the list
        # Common terms indicating T&C links
        terms_indicators = [
            'terms',
            'conditions',
            # 'tos',
            'terms of service',
            'terms of use',
            'legal',
            # 'privacy',
            # 'policy',
            'terms and conditions',
            'eula',
            'agb',
            'agreement',
            'anvandarvillkor',
        ]

        for link in soup.find_all('a'):
            href = link.get('href')
            if not href:
                continue

            full_url = urljoin(base_url, href)

            # link_text = link.get_text().lower().strip()

            if any(term in full_url for term in terms_indicators):
                logger.info(f"Found potential T&C link: {full_url}")
                terms_urls.append(full_url)

        # TODO: Specifically look for footer links, as T&C are often there

        return terms_urls

    except Exception as e:
        logger.error(f'Error extracting T&C URLs from HTML: {e}')
        return []


async def search_terms_with_google_cse(domain: str) -> List[str]:
    """Searches for T&C pages using Google Custom Search API.

    Args:
        domain: The domain to search for T&C pages.

    Returns:
        A list of potential T&C URLs found via Google CSE.
    """
    if not settings.google_cse_api_key or not settings.google_cse_id:
        logger.warning('Google CSE API key or ID not configured')
        return []

    logger.info(f'Searching for T&C pages for domain {domain} using Google CSE')

    # TODO: configure search engine Query Enhancement from google platform settings
    search_terms = [
        'terms and conditions',
    ]
    search_results = []

    for term in search_terms:
        query = f'site:{domain} {term}'
        url = (
            f'https://www.googleapis.com/customsearch/v1'
            f'?key={settings.google_cse_api_key}'
            f'&cx={settings.google_cse_id}'
            f'&q={query}'
        )

        # Use HTTP-only fetch for Google CSE JSON API
        response = await _fetch_text_http_only(url)

        if response.is_success and response.content:
            try:
                data = response.content
                results = json.loads(data)

                if 'items' in results:
                    for item in results['items']:
                        search_results.append(item['link'])

                logger.info(
                    f"Found {len(results.get('items', []))} results for query '{query}'"
                )

            except Exception as e:
                logger.error(f'Error parsing Google CSE results: {e}')
        else:
            logger.warning(f"Failed to get Google CSE results for query '{query}'")

    return search_results

# TODO: if needed to decode cloudflare hashed protected emails
# def decode_cf_email(protected_hex):
#     # first byte is the XOR key
#     key = int(protected_hex[:2], 16)
#     # decode each subsequent byte by XOR’ing with key
#     chars = [
#         chr(int(protected_hex[i:i+2], 16) ^ key)
#         for i in range(2, len(protected_hex), 2)
#     ]
#     return "".join(chars)

async def fetch_related_legal_document(url: str, link_text: str) -> RelatedLegalDocument:
    """Fetches and processes a related legal document referenced in the main T&C.

    This function fetches a document referenced by a link in the main T&C page,
    handling both binary and text content appropriately. For text content,
    it extracts the main text using trafilatura.

    Args:
        url: The URL of the related document to fetch.
        link_text: The text of the link referencing this document.

    Returns:
        A RelatedLegalDocument object containing the fetched and processed document.
    """
    logger.info(f'Fetching related legal document: "{link_text}" from {url}')

    related_doc = RelatedLegalDocument(link_text=link_text, url=url)

    try:
        # Fetch the related document
        fetched_content = await fetch_url(url)

        if fetched_content.is_success:
            if fetched_content.is_binary:
                related_doc.is_binary = True
                related_doc.content_type = fetched_content.content_type
                related_doc.binary_size = fetched_content.binary_content_size
                related_doc.storage_uri = fetched_content.storage_uri
                related_doc.etag = fetched_content.etag
                related_doc.last_modified = fetched_content.last_modified

                logger.info(
                    f'Successfully fetched binary related document: {link_text}. '
                    f'Type: {fetched_content.content_type}, '
                    f'Size: {fetched_content.binary_content_size} bytes'
                )

            elif fetched_content.content:
                extracted_text, _ = extract_text_trafilatura(fetched_content.content)
                related_doc.content = extracted_text
                related_doc.content_type = fetched_content.content_type

                if extracted_text:
                    related_doc.content_hash = hashlib.sha256(
                        extracted_text.encode('utf-8')
                    ).hexdigest()

                logger.info(
                    f'Successfully fetched and extracted text from related document: {link_text}'
                )
            else:
                related_doc.fetch_error = 'Fetch successful but no content available'
                logger.warning(f'No content available in related document: {url}')
        else:
            related_doc.fetch_error = fetched_content.error or 'Unknown fetch error'
            logger.warning(
                f'Failed to fetch related document {url}: {related_doc.fetch_error}'
            )

    except Exception as e:
        error_msg = f'Error processing related document {url}: {str(e)}'
        related_doc.fetch_error = error_msg
        logger.error(error_msg)

    return related_doc


async def fetch_terms_content(terms_url: str) -> FetchedContent:
    """Fetches and extracts the content of a T&C page.

    This function handles both HTML/text T&C pages and binary document formats (PDF, DOC, etc.).
    For binary formats, the content is stored in S3 and a reference URI is included in the result.

    Args:
        terms_url: The URL of the T&C page to fetch.

    Returns:
        The fetched content of the T&C page.
    """
    logger.info(f'Fetching T&C content from {terms_url}')
    # For T&C URLs, reuse the HTML-first strategy with Selenium fallback,
    # enabling binary detection so we persist binary docs and skip Selenium.
    return await _fetch_html_with_fallback(terms_url, detect_binary=True)


async def process_terms_variant(
    url: str, base_url: str
    ) -> TermsVariant | None:
    """Process a single T&C URL to create a TermsVariant.

    Args:
        url: The T&C URL to process.
        base_url: The base URL for resolving relative links.

    Returns:
        A TermsVariant object if successful, None otherwise.
    """
    try:
        # Fetch the T&C content
        fetched_content = await fetch_terms_content(url)

        if not fetched_content.is_success:
            logger.warning(f'Failed to fetch T&C from {url}')
            return None

        # Create the variant
        variant = TermsVariant(fetched=fetched_content)

        # Process text content if available
        if not fetched_content.is_binary and fetched_content.content:
            # Extract text and links using trafilatura
            extracted_text, links = extract_text_trafilatura(fetched_content.content)

            if extracted_text:
                variant.extracted_text = extracted_text
                variant.extracted_text_hash = hashlib.sha256(
                    extracted_text.encode('utf-8')
                ).hexdigest()

                # Process related legal documents from links
                if links:
                    logger.info(
                        f'Found {len(links)} potential related documents in T&C'
                    )

                    # Filter for legal document links
                    legal_keywords = [
                        'agreement', 'terms',
                        'conditions', 'legal', 'dpa'
                    ]

                    related_tasks = []
                    for link in links:
                        link_text_lower = link['link_text'].lower()
                        if any(kw in link_text_lower for kw in legal_keywords):
                            # Resolve relative URLs
                            full_url = urljoin(base_url, link['url'])
                            related_tasks.append(
                                fetch_related_legal_document(
                                    full_url, link['link_text']
                                )
                            )

                    # Fetch related documents concurrently (limit concurrency)
                    if related_tasks:
                        # Process in batches of 5 to avoid overwhelming the server
                        batch_size = 5
                        for i in range(0, len(related_tasks), batch_size):
                            batch = related_tasks[i:i + batch_size]
                            batch_results = await asyncio.gather(
                                *batch, return_exceptions=True
                            )

                            for result in batch_results:
                                if isinstance(result, RelatedLegalDocument):
                                    variant.related_documents.append(result)
                                else:
                                    logger.error(
                                        f'Error fetching related document: {result}'
                                    )

        return variant

    except Exception as e:
        logger.error(f'Error processing terms variant from {url}: {e}')
        return None


async def discover_terms_artifacts(base_url: HttpUrl) -> DiscoveredArtifacts:
    """Discovers and fetches T&C pages following a prioritized decision tree.

    The discovery process follows these steps in sequence:
    0. Validate the URL with DNS resolution
    1. Fetch robots.txt and check for sitemap URLs
    2. If sitemaps exist, parse them to find T&C URLs
    3. If no T&C URLs found, fetch and extract from primary HTML
    4. If still no URLs found, try Google Custom Search
    5. As a last resort, perform a BFS crawl with limited depth
    6. Fetch and extract the content of the most promising T&C URL,
       then extract clean text and hash it.

    Args:
        base_url: The validated base URL of the source.

    Returns:
        A DiscoveredArtifacts object containing the discovered T&C information.
        If URL validation fails, the object will contain an error message.
    """
    # Step 0: Validate URL with DNS resolution
    valid, error_msg = validate_url(base_url)

    # Initialize artifacts container
    artifacts = DiscoveredArtifacts(base_url=base_url)

    # If URL validation fails, set error and return early
    if not valid:
        logger.warning(f'URL validation failed for {base_url}: {error_msg}')
        artifacts.error = f'URL validation failed: {error_msg}'
        return artifacts

    discovered_tc_urls: List[str] = []

    try:
        # Step 1: Fetch only robots.txt first, following the decision tree approach
        logger.info(f'Fetching robots.txt for {base_url}')
        robots_fetch = await fetch_robots_txt(base_url)
        artifacts.robots_txt = robots_fetch.content
        # html_fetch = None

        if robots_fetch.is_success and robots_fetch.content:
            logger.info(f'Successfully fetched robots.txt for {base_url}')
            sitemap_urls = extract_sitemap_urls_from_robots(robots_fetch)

            # Filter to only include XML sitemaps exclude RSS and other formats
            xml_sitemap_urls = [
                url
                for url in sitemap_urls
                if url.lower().endswith('.xml') and not url.lower().endswith('rss.xml')
            ]

            if xml_sitemap_urls:
                logger.info(f'Found XML sitemap URLs: {xml_sitemap_urls}')

                all_sitemap_page_urls: List[str] = []
                for s_url in xml_sitemap_urls:
                    sitemap_page_urls = await parse_sitemap(s_url)
                    all_sitemap_page_urls.extend(sitemap_page_urls)

                for page_url in all_sitemap_page_urls:
                    if is_terms_url(page_url):
                        discovered_tc_urls.append(page_url)

                if discovered_tc_urls:
                    logger.info(f'Found T&C URLs via sitemaps: {discovered_tc_urls}')
                    artifacts.discovery_method = DiscoveryMethod.SITEMAP
        else:
            logger.warning(f'Failed to fetch or parse robots.txt for {base_url}')

        # Step 3: If no T&C URLs from sitemaps, fetch and try primary HTML
        if not discovered_tc_urls:
            logger.info(
                f'No T&C URLs found from sitemaps, fetching primary HTML for {base_url}'
            )
            html_fetch = await fetch_primary_html(base_url)
            artifacts.primary_html = html_fetch

            if html_fetch.is_success and html_fetch.content:
                logger.info(
                    f'Analyzing primary HTML from {html_fetch.final_url or html_fetch.request_url}'
                )
                html_terms_urls = extract_terms_urls_from_html(html_fetch, str(base_url))
                if html_terms_urls:
                    discovered_tc_urls.extend(html_terms_urls)
                    logger.info(f'Found T&C URLs via HTML: {html_terms_urls}')
                    artifacts.discovery_method = DiscoveryMethod.HTML_EXTRACTION

        # Step 4: If still no URLs, try Google Custom Search
        if not discovered_tc_urls and settings.google_cse_id and settings.google_cse_api_key:
            logger.info(f'No T&C URLs yet, trying Google CSE for domain: {base_url.host}')
            try:
                cse_terms_urls = await search_terms_with_google_cse(str(base_url.host))
                if cse_terms_urls:
                    discovered_tc_urls.extend(cse_terms_urls)
                    logger.info(f'Found T&C URLs via Google CSE: {cse_terms_urls}')
                    artifacts.discovery_method = DiscoveryMethod.GOOGLE_CSE
            except Exception as e:
                logger.warning(f'Google CSE search failed: {e}')

        artifacts.terms_urls = list(set(discovered_tc_urls))  # Deduplicate

        # Filter Google CSE results to keep only URLs that likely correspond to T&C pages
        if artifacts.discovery_method == DiscoveryMethod.GOOGLE_CSE:
            google_tc_urls = [u for u in artifacts.terms_urls if is_terms_url(u)]
            if google_tc_urls:
                logger.info(
                    f'Filtered {len(artifacts.terms_urls)} Google CSE results down to '
                    f'{len(google_tc_urls)} likely T&C URLs'
                )
                artifacts.terms_urls = google_tc_urls
            else:
                logger.info(
                    'No Google CSE results matched T&C heuristic; no valid candidate URLs found.'
                )
                artifacts.terms_urls = []
                artifacts.discovery_method = None

        # Step 5: Select T&C URL and fetch its content
        if artifacts.terms_urls:
            logger.info(
                f'Processing {len(artifacts.terms_urls)} discovered T&C URLs'
            )
            artifacts.term_variants = []
            artifacts_error_buf: list[str] = []

            # process T&C urls concurrently with limited concurrency
            async def _process_with_semaphore(url: str) -> TermsVariant | None:
                async with get_fetch_semaphore():
                    return await process_terms_variant(url, str(base_url))

            variant_results = await asyncio.gather(
                *[_process_with_semaphore(url) for url in artifacts.terms_urls],
                return_exceptions=True
            )

            for i, result in enumerate(variant_results):
                url = artifacts.terms_urls[i]
                if isinstance(result, Exception):
                    error_msg = f'Exception processing {url}: {result}'
                    artifacts_error_buf.append(error_msg)
                    logger.error(error_msg)
                elif result is None:
                    error_msg = f'Failed to process {url}'
                    artifacts_error_buf.append(error_msg)
                    logger.error(error_msg)
                else:
                    artifacts.term_variants.append(result)
                    logger.info(f'Successfully processed T&C variant from {url}')

            if artifacts_error_buf and not artifacts.term_variants:
                artifacts.error = (
                    'Failed to process any T&C URLs. Errors: '
                    + '; '.join(artifacts_error_buf[:3])  # Limit error messages
                )
            elif artifacts_error_buf:
                logger.warning(
                    f'Processed {len(artifacts.term_variants)} T&C variants, '
                    f'{len(artifacts_error_buf)} failed'
                )

        else:
            # set terms_found to False to indicate no T&C was found
            # artifacts.terms_found = False
            logger.info(f'No T&C URLs discovered for {base_url}')
            artifacts.error = (
                artifacts.error or 'No T&C URLs discovered after all methods.'
            )
            artifacts.discovery_method = None

    except Exception as e:
        logger.error(f'Error during T&C discovery for {base_url}: {e}', exc_info=True)
        artifacts.error = str(e)

    return artifacts
