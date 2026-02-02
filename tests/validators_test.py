import socket
from unittest.mock import patch

import pytest
from pydantic import HttpUrl, ValidationError

from src.acc.helpers.validators import validate_url


def prepare_url(url_str: str) -> HttpUrl | None:
    """Adds 'https://' scheme if missing and parses URL. Returns None on syntax error."""
    if not isinstance(url_str, str):
        url_str = str(url_str)

    test_url_str = url_str
    # Add default scheme if none is present for Pydantic validation
    if '://' not in url_str:
        test_url_str = f'https://{url_str}'

    try:
        return HttpUrl(test_url_str)
    except ValidationError:
        return None  # Indicates a syntax error


DNS_TEST_CASES = [
    # Valid DNS resolution cases
    ('google.com', (True, None), None),  # Mock returns successfully (side_effect=None)
    ('http://localhost', (True, None), None),
    ('https://1.1.1.1', (True, None), None),
    ('example.com', (True, None), None),  # Scheme added by prepare_url
    # Invalid DNS resolution cases
    (
        'invalid-domain-that-does-not-exist-abcdef.xyz',
        (
            False,
            'URL DNS resolution failed for hostname: invalid-domain-that-does-not-exist-abcdef.xyz',
        ),
        socket.gaierror,
    ),  # Mock raises gaierror
    (
        'https://another-really-nonexistent.domain',
        (
            False,
            'URL DNS resolution failed for hostname: another-really-nonexistent.domain',
        ),
        socket.gaierror,
    ),
    # Cases where Pydantic should fail (tested implicitly by prepare_url returning None)
    (
        'http://',
        (False, 'Invalid URL format'),
        ValidationError,
    ),  # prepare_url returns None
    (
        'invalid-url-syntax',
        (False, 'URL DNS resolution failed for hostname: invalid-url-syntax'),
        socket.gaierror,
    ),  # prepare_url adds scheme, DNS fails
    (
        'ftp://example.com',
        (False, 'Invalid URL scheme'),
        ValidationError,
    ),  # prepare_url returns None
]


@pytest.mark.parametrize(
    'input_url_str, expected_result, mock_side_effect', DNS_TEST_CASES
)
@patch('src.acc.utils.validator.socket.gethostbyname')  # Patch socket where it's USED
def test_validate_url_dns(
    mock_gethostbyname, input_url_str: str, expected_result: tuple, mock_side_effect
):
    """Tests the validate_url function, mocking DNS lookups.

    Covers both valid/invalid DNS scenarios and handles implicit syntax checks.
    """
    # Configure the mock's behavior based on the test case
    # If the side effect is an Exception class, the mock will raise it when called.
    # If it's None, the mock will return its default value (MagicMock instance), simulating success.
    mock_gethostbyname.side_effect = mock_side_effect

    # Prepare the URL (handles scheme addition and basic syntax validation)
    parsed_url = prepare_url(input_url_str)

    # If prepare_url returns None, it means syntax was invalid - check if this matches expected
    if parsed_url is None:
        assert (
            expected_result[0] is False
        ), f"Expected syntax error for '{input_url_str}', but test case expected success."
        # Optionally check if the test case expected a ValidationError side effect
        assert (
            mock_side_effect is ValidationError
        ), f"Expected ValidationError for '{input_url_str}' syntax error, but got {mock_side_effect}"
        return
    else:
        # If syntax is okay, but test case expected failure (e.g., expected ValidationError), fail the test
        if mock_side_effect is ValidationError:
            pytest.fail(
                f"Expected syntax error (ValidationError) for '{input_url_str}', but prepare_url succeeded."
            )

    # Call the actual function under test
    is_valid, error_msg = validate_url(parsed_url)

    # Assert Results
    expected_bool, expected_msg_start = expected_result
    assert is_valid == expected_bool, f"Validation status mismatch for '{input_url_str}'"

    if expected_msg_start:
        assert (
            error_msg is not None
        ), f"Expected error message for '{input_url_str}' but got None"
        assert error_msg.startswith(
            expected_msg_start
        ), f"Error message mismatch for '{input_url_str}'. Expected start: '{expected_msg_start}', Got: '{error_msg}'"
    else:
        assert (
            error_msg is None
        ), f"Expected no error message for '{input_url_str}' but got '{error_msg}'"

    # Assert Mock Calls
    # Mock should only be called if DNS resolution was attempted (i.e., not a syntax error caught by prepare_url
    # and the hostname exists, and the side_effect wasn't ValidationError).
    if parsed_url and parsed_url.host and mock_side_effect is not ValidationError:
        # Expect a call if the mock was configured to return successfully (side_effect=None)
        # or if it was configured to raise gaierror (side_effect=socket.gaierror)
        if mock_side_effect is None or mock_side_effect is socket.gaierror:
            mock_gethostbyname.assert_called_once_with(parsed_url.host)
        else:
            # Should not be called for other side effects if any were defined
            mock_gethostbyname.assert_not_called()
    else:
        # Should not be called if syntax was invalid or host missing
        mock_gethostbyname.assert_not_called()
