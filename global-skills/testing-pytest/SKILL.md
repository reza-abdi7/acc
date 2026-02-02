---
name: testing-pytest
description: Comprehensive pytest testing patterns including fixtures, async testing, mocking, parametrization, and coverage. Use when writing, reviewing, or debugging tests.
---

# Testing with Pytest

Comprehensive guide for writing effective tests using pytest, covering fixtures, async testing, mocking, and best practices.

## When to Use This Skill

- Writing unit or integration tests
- Setting up test fixtures and factories
- Testing async code
- Mocking external dependencies
- Debugging failing tests
- Improving test coverage

## Project Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── conftest.py          # Unit-specific fixtures
│   ├── test_models.py
│   └── test_services.py
├── integration/
│   ├── __init__.py
│   ├── conftest.py          # Integration-specific fixtures
│   ├── test_api.py
│   └── test_database.py
└── e2e/
    └── test_workflows.py
```

## Pytest Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
python_classes = ["Test*"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
    "-ra",
]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

## Fixtures

### Basic Fixtures

```python
# conftest.py
import pytest
from datetime import datetime

@pytest.fixture
def sample_url():
    """Simple fixture returning a value."""
    return "https://example.com"

@pytest.fixture
def sample_source():
    """Fixture returning a dict."""
    return {
        "url": "https://example.com",
        "domain": "example.com",
        "created_at": datetime.utcnow(),
    }

# Usage in test
def test_source_domain(sample_source):
    assert sample_source["domain"] == "example.com"
```

### Fixture Scopes

```python
@pytest.fixture(scope="function")  # Default: new for each test
def function_fixture():
    return create_resource()

@pytest.fixture(scope="class")  # Shared within test class
def class_fixture():
    return create_resource()

@pytest.fixture(scope="module")  # Shared within module
def module_fixture():
    return create_resource()

@pytest.fixture(scope="session")  # Shared across entire test session
def session_fixture():
    return create_expensive_resource()
```

### Fixtures with Setup and Teardown

```python
@pytest.fixture
def temp_file(tmp_path):
    """Fixture with cleanup."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("test content")
    yield file_path  # Test runs here
    # Cleanup after test
    if file_path.exists():
        file_path.unlink()

@pytest.fixture
def db_session():
    """Database session with rollback."""
    session = create_session()
    yield session
    session.rollback()
    session.close()
```

### Fixture Factories

```python
@pytest.fixture
def source_factory():
    """Factory fixture for creating test sources."""
    created_sources = []
    
    def _create_source(url: str = "https://example.com", **kwargs):
        source = Source(url=url, **kwargs)
        created_sources.append(source)
        return source
    
    yield _create_source
    
    # Cleanup
    for source in created_sources:
        source.delete()

# Usage
def test_multiple_sources(source_factory):
    source1 = source_factory("https://site1.com")
    source2 = source_factory("https://site2.com", status="active")
    assert source1.url != source2.url
```

## Async Testing

### Basic Async Tests

```python
import pytest

# With asyncio_mode = "auto" in config, no decorator needed
async def test_async_function():
    result = await some_async_function()
    assert result == expected

# Or explicitly mark
@pytest.mark.asyncio
async def test_explicit_async():
    result = await fetch_data()
    assert result is not None
```

### Async Fixtures

```python
@pytest.fixture
async def async_client():
    """Async HTTP client fixture."""
    async with httpx.AsyncClient() as client:
        yield client

@pytest.fixture
async def db_session():
    """Async database session."""
    async with AsyncSession(engine) as session:
        yield session
        await session.rollback()

# Usage
async def test_fetch_url(async_client):
    response = await async_client.get("https://example.com")
    assert response.status_code == 200
```

### Testing FastAPI

```python
import pytest
from httpx import AsyncClient, ASGITransport
from src.acc.api.main import app

@pytest.fixture
async def api_client():
    """FastAPI test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

async def test_health_endpoint(api_client):
    response = await api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

async def test_create_source(api_client):
    response = await api_client.post(
        "/api/v1/sources",
        json={"url": "https://example.com"}
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
```

## Mocking

### Basic Mocking with pytest-mock

```python
def test_with_mock(mocker):
    """Using pytest-mock's mocker fixture."""
    # Mock a function
    mock_fetch = mocker.patch("src.acc.fetcher.fetch_url")
    mock_fetch.return_value = "<html>content</html>"
    
    result = process_url("https://example.com")
    
    mock_fetch.assert_called_once_with("https://example.com")
    assert "content" in result
```

### Mocking Async Functions

```python
async def test_async_mock(mocker):
    """Mocking async functions."""
    # Create async mock
    mock_fetch = mocker.patch(
        "src.acc.fetcher.fetch_url",
        new_callable=mocker.AsyncMock
    )
    mock_fetch.return_value = {"status": "success"}
    
    result = await process_async("https://example.com")
    
    mock_fetch.assert_awaited_once()
    assert result["status"] == "success"
```

### Mocking Context Managers

```python
async def test_mock_context_manager(mocker):
    """Mocking async context managers."""
    mock_session = mocker.MagicMock()
    mock_session.__aenter__ = mocker.AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = mocker.AsyncMock(return_value=None)
    
    mocker.patch("src.acc.db.get_session", return_value=mock_session)
    
    await some_db_operation()
    
    mock_session.__aenter__.assert_awaited_once()
```

### Mocking External Services

```python
@pytest.fixture
def mock_http_responses(mocker):
    """Mock HTTP responses."""
    responses = {}
    
    async def mock_get(url, **kwargs):
        if url in responses:
            mock_response = mocker.MagicMock()
            mock_response.status_code = 200
            mock_response.text = responses[url]
            mock_response.json.return_value = responses.get(f"{url}_json", {})
            return mock_response
        raise httpx.HTTPError(f"No mock for {url}")
    
    mock_client = mocker.patch("httpx.AsyncClient.get", side_effect=mock_get)
    
    return responses

async def test_fetch_terms(mock_http_responses):
    mock_http_responses["https://example.com/terms"] = "<html>Terms</html>"
    
    result = await fetch_terms("https://example.com")
    assert "Terms" in result
```

## Parametrization

### Basic Parametrization

```python
@pytest.mark.parametrize("url,expected", [
    ("https://example.com", True),
    ("http://example.com", True),
    ("ftp://example.com", False),
    ("not-a-url", False),
])
def test_is_valid_url(url, expected):
    assert is_valid_url(url) == expected
```

### Multiple Parameters

```python
@pytest.mark.parametrize("status,confidence,expected_decision", [
    ("ALLOWED", 0.95, "auto_approve"),
    ("ALLOWED", 0.60, "review_required"),
    ("BLOCKED", 0.95, "auto_reject"),
    ("BLOCKED", 0.60, "review_required"),
])
def test_decision_logic(status, confidence, expected_decision):
    result = make_decision(status, confidence)
    assert result == expected_decision
```

### Parametrize with IDs

```python
@pytest.mark.parametrize("input_data,expected", [
    pytest.param(
        {"url": "https://example.com"},
        {"valid": True},
        id="valid-https-url"
    ),
    pytest.param(
        {"url": ""},
        {"valid": False},
        id="empty-url"
    ),
    pytest.param(
        {"url": None},
        {"valid": False},
        id="null-url"
    ),
])
def test_validate_input(input_data, expected):
    result = validate(input_data)
    assert result["valid"] == expected["valid"]
```

### Combining Parametrize

```python
@pytest.mark.parametrize("method", ["GET", "POST"])
@pytest.mark.parametrize("status_code", [200, 404, 500])
async def test_http_handling(api_client, method, status_code):
    """Tests all combinations: GET/200, GET/404, GET/500, POST/200, etc."""
    # Test implementation
    pass
```

## Test Organization

### Test Classes

```python
class TestSourceValidation:
    """Group related tests."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Runs before each test in class."""
        self.validator = SourceValidator()
    
    def test_valid_url(self):
        assert self.validator.is_valid("https://example.com")
    
    def test_invalid_url(self):
        assert not self.validator.is_valid("not-a-url")
    
    def test_empty_url(self):
        assert not self.validator.is_valid("")
```

### Markers for Test Selection

```python
@pytest.mark.unit
def test_fast_unit_test():
    pass

@pytest.mark.integration
def test_database_integration():
    pass

@pytest.mark.slow
def test_slow_operation():
    pass

# Run specific markers
# pytest -m unit
# pytest -m "not slow"
# pytest -m "integration and not slow"
```

## Assertions and Matchers

### Built-in Assertions

```python
def test_assertions():
    # Equality
    assert result == expected
    
    # Truthiness
    assert value
    assert not empty_value
    
    # Membership
    assert item in collection
    assert key in dictionary
    
    # Type checking
    assert isinstance(obj, MyClass)
    
    # Approximate equality (floats)
    assert result == pytest.approx(3.14, rel=1e-2)
```

### Exception Testing

```python
def test_raises_exception():
    with pytest.raises(ValueError) as exc_info:
        validate_url("invalid")
    
    assert "Invalid URL" in str(exc_info.value)

def test_raises_with_match():
    with pytest.raises(ValueError, match=r"Invalid.*URL"):
        validate_url("invalid")

async def test_async_raises():
    with pytest.raises(FetchError):
        await fetch_url("https://nonexistent.invalid")
```

### Custom Assertions

```python
def assert_valid_compliance_decision(decision):
    """Custom assertion helper."""
    assert decision is not None
    assert decision.status in ["ALLOWED", "BLOCKED", "CONDITIONAL", "REVIEW_REQUIRED"]
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.source_id is not None

def test_compliance_decision(decision_factory):
    decision = decision_factory()
    assert_valid_compliance_decision(decision)
```

## Coverage

```toml
# pyproject.toml
[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "*/tests/*",
    "*/__init__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
fail_under = 80
show_missing = true
```

```bash
# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term-missing

# Generate report
coverage html
```

## Debugging Tests

```python
# Use breakpoint
def test_debug():
    result = complex_function()
    breakpoint()  # Drops into debugger
    assert result == expected

# Print with -s flag
def test_with_print():
    print(f"Debug: {variable}")  # pytest -s to see output

# Use pytest's capsys
def test_capture_output(capsys):
    print("Hello")
    captured = capsys.readouterr()
    assert "Hello" in captured.out
```

```bash
# Useful pytest flags
pytest -v                    # Verbose
pytest -s                    # Show print statements
pytest -x                    # Stop on first failure
pytest --pdb                 # Drop into debugger on failure
pytest --lf                  # Run last failed tests
pytest -k "test_name"        # Run tests matching pattern
pytest --tb=long             # Full tracebacks
```

## Best Practices

### DO:
- **Arrange-Act-Assert**: Structure tests clearly
- **One assertion concept per test**: Test one thing
- **Descriptive names**: `test_fetch_returns_none_for_invalid_url`
- **Use fixtures**: Share setup, avoid duplication
- **Test edge cases**: Empty, None, boundaries
- **Mock external services**: Tests should be isolated

### DON'T:
- **Test implementation details**: Test behavior, not internals
- **Share state between tests**: Each test should be independent
- **Use sleep in tests**: Use proper async waiting
- **Ignore flaky tests**: Fix or remove them
- **Over-mock**: If everything is mocked, what are you testing?
