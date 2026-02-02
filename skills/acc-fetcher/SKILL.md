---
name: acc-fetcher
description: Web scraping and document fetching patterns for T&C, robots.txt, and LLM.txt extraction. Use when implementing or debugging the Fetcher component.
---

# ACC Fetcher

The Fetcher component is responsible for retrieving legal documents (T&C, robots.txt, LLM.txt) from external web sources and storing them with appropriate metadata.

## When to Use This Skill

- Implementing document fetching logic
- Adding support for new document types
- Debugging fetch failures
- Handling edge cases (encoding, redirects, JavaScript-rendered content)
- Optimizing fetch performance

## Fetcher Responsibilities

1. **URL Resolution**: Normalize and resolve URLs to canonical form
2. **Document Discovery**: Find T&C, robots.txt, LLM.txt locations
3. **Content Retrieval**: Fetch documents via HTTP(S)
4. **Content Extraction**: Extract text from HTML/other formats
5. **Metadata Collection**: Language, geo, timestamps
6. **Storage**: Persist raw and processed content to DB

## Document Types

### 1. Terms & Conditions (T&C)

**Locations** (common patterns):
```
/terms
/terms-of-service
/terms-and-conditions
/tos
/legal/terms
/legal
/terms-of-use
```

**Challenges**:
- Often JavaScript-rendered
- May require cookie acceptance
- Multiple languages/regions
- Frequently updated

### 2. robots.txt

**Location**: Always at root
```
https://example.com/robots.txt
```

**Format**: Standard robots exclusion protocol
```
User-agent: *
Disallow: /private/
Allow: /public/

User-agent: GPTBot
Disallow: /
```

### 3. LLM.txt

**Location**: Root or well-known
```
https://example.com/llm.txt
https://example.com/.well-known/llm.txt
```

**Purpose**: AI/LLM-specific usage permissions (emerging standard)

## Implementation Patterns

### HTTP Client Setup

```python
import httpx
from typing import Optional

class FetcherClient:
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            max_redirects=5,
            http2=True,
            headers={
                "User-Agent": "ACC-Compliance-Checker/1.0 (+https://company.com/acc)",
                "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
    
    async def fetch(self, url: str) -> FetchResult:
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            return FetchResult(
                url=str(response.url),  # Final URL after redirects
                status_code=response.status_code,
                content=response.text,
                content_type=response.headers.get("content-type"),
                encoding=response.encoding,
                headers=dict(response.headers),
            )
        except httpx.TimeoutException:
            raise FetchError(url, "timeout")
        except httpx.HTTPStatusError as e:
            raise FetchError(url, f"HTTP {e.response.status_code}")
```

### URL Normalization

```python
import tldextract
from urllib.parse import urlparse, urljoin

def normalize_url(url: str) -> str:
    """Normalize URL to canonical form."""
    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    
    parsed = urlparse(url)
    
    # Normalize to lowercase domain
    normalized = parsed._replace(
        netloc=parsed.netloc.lower(),
        path=parsed.path.rstrip("/") or "/",
    )
    
    return normalized.geturl()

def extract_domain(url: str) -> DomainInfo:
    """Extract domain components."""
    ext = tldextract.extract(url)
    return DomainInfo(
        subdomain=ext.subdomain,
        domain=ext.domain,
        suffix=ext.suffix,
        registered_domain=ext.registered_domain,  # e.g., "example.com"
    )

def get_robots_url(url: str) -> str:
    """Get robots.txt URL for a given URL."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/robots.txt"
```

### T&C Discovery

```python
from bs4 import BeautifulSoup

# Common T&C link patterns
TC_LINK_PATTERNS = [
    r"terms",
    r"tos",
    r"legal",
    r"conditions",
    r"nutzungsbedingungen",  # German
    r"agb",  # German
    r"条款",  # Chinese
]

TC_PATHS = [
    "/terms",
    "/terms-of-service",
    "/terms-and-conditions",
    "/tos",
    "/legal/terms",
    "/legal",
]

async def discover_tc_url(base_url: str, html_content: str) -> Optional[str]:
    """Discover T&C URL from page content or common paths."""
    soup = BeautifulSoup(html_content, "lxml")
    
    # Look for links in footer or with relevant text
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        text = link.get_text(strip=True).lower()
        
        for pattern in TC_LINK_PATTERNS:
            if pattern in href.lower() or pattern in text:
                return urljoin(base_url, href)
    
    # Try common paths
    for path in TC_PATHS:
        tc_url = urljoin(base_url, path)
        if await url_exists(tc_url):
            return tc_url
    
    return None
```

### Content Extraction

```python
from trafilatura import extract
from bs4 import BeautifulSoup

def extract_text_content(html: str, url: str) -> ExtractedContent:
    """Extract main text content from HTML."""
    # Try trafilatura first (good for article-like content)
    text = extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    
    if text:
        return ExtractedContent(
            text=text,
            method="trafilatura",
        )
    
    # Fallback to BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    
    # Remove script and style elements
    for element in soup(["script", "style", "nav", "header", "footer"]):
        element.decompose()
    
    # Get text
    text = soup.get_text(separator="\n", strip=True)
    
    return ExtractedContent(
        text=text,
        method="beautifulsoup",
    )
```

### JavaScript-Rendered Content

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

async def fetch_js_rendered(url: str) -> str:
    """Fetch JavaScript-rendered content using Selenium."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    # Use remote Selenium if configured
    driver = webdriver.Remote(
        command_executor=settings.selenium_remote_url,
        options=options,
    )
    
    try:
        driver.get(url)
        
        # Wait for content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(("tag name", "body"))
        )
        
        # Additional wait for JS rendering
        await asyncio.sleep(2)
        
        return driver.page_source
    finally:
        driver.quit()
```

### Language Detection

```python
from langdetect import detect, detect_langs

def detect_language(text: str) -> LanguageInfo:
    """Detect language of text content."""
    try:
        # Get primary language
        primary = detect(text)
        
        # Get language probabilities
        langs = detect_langs(text)
        probabilities = {str(lang.lang): lang.prob for lang in langs}
        
        return LanguageInfo(
            primary=primary,
            probabilities=probabilities,
            confidence=probabilities.get(primary, 0.0),
        )
    except Exception:
        return LanguageInfo(
            primary="unknown",
            probabilities={},
            confidence=0.0,
        )
```

### Robots.txt Parsing

```python
from urllib.robotparser import RobotFileParser

def parse_robots_txt(content: str, base_url: str) -> RobotsInfo:
    """Parse robots.txt content."""
    parser = RobotFileParser()
    parser.parse(content.splitlines())
    
    # Check permissions for various user agents
    user_agents = ["*", "Googlebot", "GPTBot", "CCBot", "anthropic-ai"]
    
    permissions = {}
    for ua in user_agents:
        permissions[ua] = {
            "can_fetch_root": parser.can_fetch(ua, "/"),
            "crawl_delay": parser.crawl_delay(ua),
        }
    
    # Extract all disallow rules
    disallow_rules = []
    for line in content.splitlines():
        if line.lower().startswith("disallow:"):
            path = line.split(":", 1)[1].strip()
            if path:
                disallow_rules.append(path)
    
    return RobotsInfo(
        raw_content=content,
        permissions=permissions,
        disallow_rules=disallow_rules,
        has_ai_restrictions=any(
            ua in content.lower() 
            for ua in ["gptbot", "ccbot", "anthropic", "ai"]
        ),
    )
```

## Data Models

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class FetchResult(BaseModel):
    url: str
    final_url: str  # After redirects
    status_code: int
    content: str
    content_type: Optional[str]
    encoding: Optional[str]
    fetch_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
class DocumentMetadata(BaseModel):
    source_url: str
    document_type: Literal["tc", "robots", "llm"]
    language: Optional[str]
    language_confidence: float = 0.0
    geo_region: Optional[str]  # Detected or from URL
    content_hash: str  # For change detection
    fetch_timestamp: datetime
    
class StoredDocument(BaseModel):
    id: str
    metadata: DocumentMetadata
    raw_content: str
    extracted_text: str
    extraction_method: str
    version: int  # Incremented on content change
```

## Error Handling

```python
class FetchError(Exception):
    """Base fetch error."""
    def __init__(self, url: str, reason: str, retryable: bool = True):
        self.url = url
        self.reason = reason
        self.retryable = retryable
        super().__init__(f"Failed to fetch {url}: {reason}")

class DocumentNotFoundError(FetchError):
    """Document doesn't exist (404)."""
    def __init__(self, url: str):
        super().__init__(url, "not found", retryable=False)

class RateLimitError(FetchError):
    """Rate limited by server."""
    def __init__(self, url: str, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(url, "rate limited", retryable=True)

class ContentExtractionError(FetchError):
    """Failed to extract content."""
    def __init__(self, url: str, reason: str):
        super().__init__(url, f"extraction failed: {reason}", retryable=False)
```

## Retry Strategy

```python
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

def is_retryable(exception: Exception) -> bool:
    if isinstance(exception, FetchError):
        return exception.retryable
    return isinstance(exception, (httpx.TimeoutException, httpx.NetworkError))

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception(is_retryable),
)
async def fetch_with_retry(url: str) -> FetchResult:
    return await fetcher.fetch(url)
```

## Storage Pattern

```python
async def store_document(
    session: AsyncSession,
    source_id: str,
    document_type: str,
    fetch_result: FetchResult,
) -> StoredDocument:
    """Store fetched document with metadata."""
    # Extract text
    extracted = extract_text_content(fetch_result.content, fetch_result.url)
    
    # Detect language
    language_info = detect_language(extracted.text)
    
    # Calculate content hash for change detection
    content_hash = hashlib.sha256(extracted.text.encode()).hexdigest()
    
    # Check if content changed
    existing = await get_latest_document(session, source_id, document_type)
    version = 1
    if existing:
        if existing.content_hash == content_hash:
            # No change, return existing
            return existing
        version = existing.version + 1
    
    # Create new document record
    document = StoredDocument(
        id=generate_id(),
        metadata=DocumentMetadata(
            source_url=fetch_result.url,
            document_type=document_type,
            language=language_info.primary,
            language_confidence=language_info.confidence,
            content_hash=content_hash,
            fetch_timestamp=fetch_result.fetch_timestamp,
        ),
        raw_content=fetch_result.content,
        extracted_text=extracted.text,
        extraction_method=extracted.method,
        version=version,
    )
    
    session.add(document)
    await session.commit()
    
    return document
```

## Best Practices

### DO:
- **Respect robots.txt**: Check before fetching
- **Use appropriate User-Agent**: Identify yourself
- **Handle rate limits**: Back off when requested
- **Store raw content**: Keep original for re-processing
- **Version documents**: Track changes over time
- **Detect language**: Store as metadata for later translation

### DON'T:
- **Ignore robots.txt**: Respect site policies
- **Fetch too aggressively**: Use delays between requests
- **Discard metadata**: Language, encoding, timestamps are valuable
- **Assume encoding**: Detect and handle properly
- **Block on JS rendering**: Use async, have timeouts

## Testing Fetcher

```python
@pytest.fixture
def mock_http_responses(mocker):
    """Mock HTTP responses for testing."""
    responses = {}
    
    async def mock_get(self, url, **kwargs):
        if url in responses:
            return MockResponse(responses[url])
        raise httpx.HTTPStatusError(
            "Not Found",
            request=None,
            response=MockResponse({"status_code": 404}),
        )
    
    mocker.patch.object(httpx.AsyncClient, "get", mock_get)
    return responses

async def test_fetch_tc(mock_http_responses):
    mock_http_responses["https://example.com/terms"] = {
        "status_code": 200,
        "content": "<html><body>Terms content</body></html>",
    }
    
    result = await fetcher.fetch_tc("https://example.com")
    assert "Terms content" in result.extracted_text
```
