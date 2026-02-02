"""Streamlit prototype app for Terms & Conditions discovery.

This app provides a simple UI to run the discovery workflow against a given
source URL using the internal function `discover_terms_artifacts`.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from pydantic import HttpUrl, TypeAdapter, ValidationError

from src.acc.fetcher.fetcher import discover_terms_artifacts

def validate_http_url(url_str: str) -> HttpUrl:
    """Validate a string as a Pydantic HttpUrl.

    Args:
        url_str: The input URL string from the user.

    Returns:
        A parsed `HttpUrl` instance if validation succeeds.

    Raises:
        ValidationError: If the provided string is not a valid HTTP(S) URL.
    """
    return TypeAdapter(HttpUrl).validate_python(url_str)


def run_discovery(url_str: str):
    """Run the async discovery flow for a given URL string.

    Args:
        url_str: The input URL string from the Streamlit UI.

    Returns:
        The `DiscoveredArtifacts` Pydantic model returned by the discovery
        pipeline.
    """
    base_url = validate_http_url(url_str)
    return asyncio.run(discover_terms_artifacts(base_url))


def _snippet(text: str | None, limit: int = 600) -> str:
    """Return a shortened preview of a larger text blob.

    Args:
        text: Input text or None.
        limit: Maximum number of characters to keep.

    Returns:
        A truncated string suitable for preview display.
    """
    if not text:
        return ""
    t = text.strip()
    return t if len(t) <= limit else t[: limit - 3] + "..."


async def run_discovery_with_timeout(url: HttpUrl, timeout: int = 60):
    """Run discovery with timeout and cancellation support.

    Args:
        url: The URL to discover terms for
        timeout: Maximum time in seconds to wait for discovery

    Returns:
        Discovery artifacts or None if cancelled/timed out
    """
    try:
        task = asyncio.create_task(discover_terms_artifacts(url))
        result = await asyncio.wait_for(task, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        st.error(f"Discovery timed out after {timeout} seconds")
        return None
    except asyncio.CancelledError:
        st.warning("Discovery was cancelled by user")
        return None
    except Exception as e:
        st.error(f"Discovery failed: {e}")
        return None


def render_artifacts(artifacts: Any) -> None:
    """Render the discovered artifacts in Streamlit.

    Args:
        artifacts: The `DiscoveredArtifacts` model instance.
    """
    st.subheader("Summary")
    st.write(
        {
            "discovery_method": artifacts.discovery_method,
            "sitemap_urls_count": len(artifacts.sitemap_urls or []),
            "terms_urls_count": len(artifacts.terms_urls or []),
            "terms_found": artifacts.terms_found,
            "error": artifacts.error,
        }
    )

    if artifacts.sitemap_urls:
        with st.expander("Sitemap URLs", expanded=False):
            for u in artifacts.sitemap_urls:
                st.write("- ", u)

    if artifacts.terms_urls:
        with st.expander("Discovered Terms URLs", expanded=True):
            for u in artifacts.terms_urls:
                st.write("- ", u)

    if artifacts.term_variants:
        st.subheader("Term Variants")
        for i, v in enumerate(artifacts.term_variants, start=1):
            with st.container(border=True):
                st.markdown(f"**Variant #{i}**")
                meta_cols = st.columns(3)
                meta_cols[0].write({
                    "status_code": v.fetched.status_code,
                    "content_type": v.content_type,
                    "is_binary": v.is_binary,
                })
                meta_cols[1].write({
                    "request_url": v.fetched.request_url,
                    "final_url": v.fetched.final_url,
                })
                meta_cols[2].write({
                    "etag": v.fetched.etag,
                    "last_modified": v.fetched.last_modified,
                })

                if v.is_binary:
                    st.info(
                        "Binary document stored",
                        icon="📄",
                    )
                    st.write({
                        "storage_uri": v.storage_uri,
                        "binary_size": v.fetched.binary_content_size,
                    })
                else:
                    with st.expander("Extracted Text (preview)", expanded=False):
                        st.write(_snippet(v.extracted_text, 1200))

                if v.related_documents:
                    with st.expander(
                        f"Related legal documents ({len(v.related_documents)})",
                        expanded=False,
                    ):
                        for rd in v.related_documents:
                            with st.container(border=True):
                                st.write({
                                    "link_text": rd.link_text,
                                    "url": rd.url,
                                    "is_binary": rd.is_binary,
                                    "content_type": rd.content_type,
                                    "storage_uri": rd.storage_uri,
                                    "binary_size": rd.binary_size,
                                    "etag": rd.etag,
                                    "last_modified": rd.last_modified,
                                    "fetch_error": rd.fetch_error,
                                })
                                if not rd.is_binary and rd.content:
                                    st.write(_snippet(rd.content, 800))

    with st.expander("Raw JSON", expanded=False):
        st.json(artifacts.model_dump(exclude_none=True))


def main() -> None:
    """Main Streamlit app entry point."""
    st.set_page_config(
        page_title="ACC Prototype",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 Automated Compliance Check - Terms Discovery")
    st.markdown(
        "Enter a source URL to discover Terms & Conditions and related legal documents."
    )

    if 'discovery_running' not in st.session_state:
        st.session_state.discovery_running = False
    if 'cancel_discovery' not in st.session_state:
        st.session_state.cancel_discovery = False

    with st.form("discovery_form"):
        url_input = st.text_input(
            "Source URL",
            placeholder="https://example.com",
            help="Enter the main website URL to analyze",
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            timeout = st.number_input(
                "Timeout (seconds)",
                min_value=10,
                max_value=300,
                value=150,
                step=10,
                help="Maximum time to wait for discovery"
            )

        submitted = st.form_submit_button(
            "🚀 Discover Terms",
            type="primary",
            disabled=st.session_state.discovery_running
        )

    if st.session_state.discovery_running:
        if st.button("🛑 Cancel Discovery", type="secondary"):
            st.session_state.cancel_discovery = True
            st.warning("Cancellation requested...")

    if submitted and url_input and not st.session_state.discovery_running:
        st.session_state.discovery_running = False
        st.session_state.cancel_discovery = False

        try:
            validated_url = validate_http_url(url_input)
            st.session_state.discovery_running = True
            st.session_state.cancel_discovery = False

            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0, text="Starting discovery...")
                status_text = st.empty()

                start_time = time.time()

                async def discovery_with_progress():
                    steps = [
                        (0.1, "Fetching robots.txt..."),
                        (0.3, "Parsing sitemaps..."),
                        (0.5, "Searching for Terms & Conditions..."),
                        (0.7, "Extracting content..."),
                        (0.9, "Processing documents...")
                    ]

                    discovery_task = asyncio.create_task(
                        discover_terms_artifacts(validated_url)
                    )

                    for progress, message in steps:
                        if st.session_state.cancel_discovery:
                            discovery_task.cancel()
                            try:
                                await discovery_task
                            except (asyncio.CancelledError, Exception):
                                pass
                            return None

                        progress_bar.progress(progress, text=message)
                        status_text.text(f"⏱️ Elapsed: {int(time.time() - start_time)}s")

                        if discovery_task.done():
                            break

                        await asyncio.sleep(0.5)

                    try:
                        result = await asyncio.wait_for(
                            discovery_task,
                            timeout=max(1, timeout - (time.time() - start_time))
                        )
                        progress_bar.progress(1.0, text="Discovery complete!")
                        return result
                    except asyncio.TimeoutError:
                        discovery_task.cancel()
                        try:
                            await asyncio.wait_for(discovery_task, timeout=2.0)
                        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                            pass
                        raise TimeoutError(f"Discovery timed out after {timeout} seconds")
                    except asyncio.CancelledError:
                        raise TimeoutError(f"Discovery was cancelled")

                try:
                    async def run_discovery():
                        try:
                            return await discovery_with_progress()
                        except asyncio.CancelledError:
                            return None

                    artifacts = asyncio.run(run_discovery())

                    print(f"DEBUG: artifacts is None: {artifacts is None}")
                    print(f"DEBUG: cancel_discovery: {st.session_state.cancel_discovery}")
                    if artifacts:
                        print(f"DEBUG: artifacts type: {type(artifacts)}")
                        print(f"DEBUG: artifacts has terms_urls: {hasattr(artifacts, 'terms_urls')}")

                    if artifacts and not st.session_state.cancel_discovery:
                        st.success(f"✅ Discovery complete in {int(time.time() - start_time)} seconds!")
                        render_artifacts(artifacts)
                    elif st.session_state.cancel_discovery:
                        st.warning("Discovery was cancelled by user")
                    elif artifacts is None:
                        st.warning("Discovery was cancelled or timed out")
                    else:
                        st.info("Discovery completed but no artifacts found")
                except TimeoutError as e:
                    st.error(str(e), icon="⏱️")
                except Exception as e:
                    st.error(f"Discovery failed: {e}", icon="⚠️")
                finally:
                    progress_bar.empty()
                    status_text.empty()

        except ValidationError as e:
            st.error(f"Invalid URL: {e}", icon="❌")
        except Exception as e:
            st.error(f"Discovery failed: {e}", icon="⚠️")
        finally:
            st.session_state.discovery_running = False
            st.session_state.cancel_discovery = False


if __name__ == "__main__":
    main()
