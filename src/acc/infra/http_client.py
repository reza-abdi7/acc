"""HTTP client and Selenium driver management."""

import asyncio
import json
from urllib.parse import urlparse

from httpx import AsyncClient, AsyncHTTPTransport, Limits
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from src.acc.config import settings

# Store clients per event loop to avoid binding issues
_clients: dict = {}


def get_http_client() -> AsyncClient:
    """Get or create an HTTP client for the current event loop."""
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        transport = AsyncHTTPTransport(retries=settings.max_retries)
        return AsyncClient(
            transport=transport,
            limits=Limits(
                max_connections=settings.max_connections,
                max_keepalive_connections=settings.max_keepalive_pool,
            ),
            follow_redirects=True,
            timeout=settings.fetch_timeout,
            headers=settings.default_headers,
            http2=True,
        )

    if loop_id not in _clients or _clients[loop_id].is_closed:
        transport = AsyncHTTPTransport(retries=settings.max_retries)
        _clients[loop_id] = AsyncClient(
            transport=transport,
            limits=Limits(
                max_connections=settings.max_connections,
                max_keepalive_connections=settings.max_keepalive_pool,
            ),
            follow_redirects=True,
            timeout=settings.fetch_timeout,
            headers=settings.default_headers,
            http2=True,
        )

    return _clients[loop_id]


def _get_chrome_options(headless: bool = False) -> Options:
    """Get configured Chrome options for anti-bot detection."""
    options = Options()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--disable-gpu")
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-features=Translate,IsolateOrigins,site-per-process")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-client-side-phishing-detection")
    options.add_argument("--metrics-recording-only")
    options.add_argument("--disable-default-apps")

    prefs = {
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
        "profile.default_content_setting_values.notifications": 2,
    }
    options.add_experimental_option("prefs", prefs)
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    if headless:
        options.add_argument("--headless=new")

    return options


def get_selenium_driver(headless: bool = False):
    """Create a new Chrome WebDriver instance for remote Selenium Grid."""
    try:
        options = _get_chrome_options(headless)

        try:
            ua_from_env = settings.default_headers.get("User-Agent")
        except Exception:
            ua_from_env = None

        if ua_from_env:
            options.add_argument(f"user-agent={ua_from_env}")

        driver = webdriver.Remote(command_executor=settings.selenium_remote_url, options=options)
        try:
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception:
            pass

        _INJECT_JS = """
            try { Object.defineProperty(navigator, 'webdriver', {get: () => undefined}); } catch (e) {}
            if (window.pendingFetches === undefined) {
                window.pendingFetches = 0;
                const _orig = window.fetch;
                window.fetch = function () {
                    window.pendingFetches++;
                    return _orig.apply(this, arguments).finally(() => window.pendingFetches--);
                };
            }
        """

        driver._post_nav_inject = True

        cdp_injected = False
        if hasattr(driver, "execute_cdp"):
            try:
                driver.execute_cdp("Page.addScriptToEvaluateOnNewDocument", {"source": _INJECT_JS})
                cdp_injected = True
            except Exception:
                pass
        elif hasattr(driver, "execute_cdp_cmd"):
            try:
                driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": _INJECT_JS})
                cdp_injected = True
            except Exception:
                pass

        if cdp_injected:
            driver._post_nav_inject = False

        return driver
    except Exception as e:
        raise RuntimeError(f"Failed to create Chrome driver: {e}") from e


def get_statuscode_from_selenium(driver, url: str) -> int | None:
    """Inspect Chrome performance logs to find the HTTP status code for url."""
    try:
        perf_entries = driver.get_log("performance")
    except Exception:
        return None

    try:
        final_url = getattr(driver, "current_url", None)
    except Exception:
        final_url = None

    def _strip_fragment(u: str | None) -> str | None:
        if not u:
            return None
        p = urlparse(u)
        path = p.path[:-1] if p.path.endswith("/") and p.path != "/" else p.path
        return f"{p.scheme}://{p.netloc}{path or ''}{('?' + p.query) if p.query else ''}"

    def _host(u: str | None) -> str | None:
        try:
            return urlparse(u).netloc.lower() if u else None
        except Exception:
            return None

    targets_exact = {t for t in (_strip_fragment(url), _strip_fragment(final_url)) if t}
    targets_host = {h for h in (_host(url), _host(final_url)) if h}

    candidates: list[tuple[int, int, int]] = []
    seq = 0

    for entry in perf_entries:
        seq += 1
        try:
            msg = json.loads(entry.get("message", "{}")).get("message", {})
            if msg.get("method") != "Network.responseReceived":
                continue

            params = msg.get("params", {})
            resp = params.get("response", {})
            status = resp.get("status")
            resp_url = resp.get("url") or ""
            resp_type = params.get("type")

            if not status or not resp_url:
                continue

            resp_url_norm = _strip_fragment(resp_url)
            resp_host = _host(resp_url)

            score = 0
            if resp_type == "Document":
                score += 5
            if resp_url_norm and resp_url_norm in targets_exact:
                score += 4
            if resp_host and resp_host in targets_host:
                score += 2

            candidates.append((score, seq, int(status)))
        except Exception:
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[-1][2]
