"""URL validation helpers."""

import logging
import socket

from pydantic import HttpUrl

logger = logging.getLogger(__name__)


def validate_url(url: HttpUrl) -> tuple[bool, str | None]:
    """Validate a URL by checking DNS resolution for its hostname.

    Args:
        url: The Pydantic HttpUrl object to validate.

    Returns:
        A tuple of (is_valid, error_message).
    """
    hostname = url.host
    if not hostname:
        msg = f"Invalid URL: Missing hostname in '{str(url)}'"
        logger.warning(msg)
        return False, msg

    try:
        socket.gethostbyname(hostname)
        logger.info(f'URL DNS resolution successful for: {hostname} ({str(url)})')
        return True, None
    except socket.gaierror:
        msg = f'URL DNS resolution failed for hostname: {hostname} ({str(url)})'
        logger.warning(msg)
        return False, msg
    except Exception as e:
        msg = f'Unexpected error during DNS resolution for {hostname} ({str(url)}): {e}'
        logger.error(msg)
        return False, msg
