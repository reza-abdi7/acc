---
name: python-best-practices
description: Python coding standards, type hints, async/await patterns, project structure, linting, and modern Python idioms. Use when writing, reviewing, or refactoring Python code.
---

# Python Best Practices

Comprehensive guidelines for writing clean, maintainable, and performant Python code following modern standards.

## When to Use This Skill

- Writing new Python code
- Reviewing or refactoring existing code
- Setting up Python project structure
- Implementing async/await patterns
- Adding type hints and documentation

## Code Style and Formatting

### General Principles

- **PEP 8**: Follow PEP 8 style guide as the baseline
- **Consistency**: Match existing codebase style when contributing
- **Readability**: Code is read more often than written

### Naming Conventions

```python
# Variables and functions: snake_case
user_name = "john"
def calculate_total_price(items: list) -> float:
    pass

# Classes: PascalCase
class UserAccount:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 30

# Private/internal: prefix with underscore
_internal_cache = {}
def _helper_function():
    pass

# "Protected" in classes: single underscore
class MyClass:
    def _protected_method(self):
        pass

# Name mangling (rarely needed): double underscore
class MyClass:
    def __private_method(self):
        pass
```

### Import Organization

```python
# Standard library imports (alphabetical)
import asyncio
import json
from collections import defaultdict
from pathlib import Path

# Third-party imports (alphabetical)
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Local imports (alphabetical)
from src.acc.config import settings
from src.acc.models import Source
```

### Line Length and Formatting

- **Max line length**: 88-100 characters (match project config)
- **Use ruff or black** for automatic formatting
- **Trailing commas** in multi-line structures for cleaner diffs

```python
# Good: trailing commas
config = {
    "host": "localhost",
    "port": 8080,
    "debug": True,  # trailing comma
}

# Good: multi-line function calls
result = some_long_function_name(
    first_argument="value",
    second_argument="another_value",
    third_argument=some_variable,
)
```

## Type Hints

### Basic Type Hints

```python
from typing import Optional, Union
from collections.abc import Sequence, Mapping

# Basic types
def greet(name: str) -> str:
    return f"Hello, {name}"

# Optional (can be None)
def find_user(user_id: int) -> Optional[User]:
    return users.get(user_id)

# Union types (Python 3.10+ use |)
def process(value: int | str) -> str:
    return str(value)

# Collections
def process_items(items: list[str]) -> dict[str, int]:
    return {item: len(item) for item in items}

# Callable
from collections.abc import Callable
def apply_func(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)
```

### Advanced Type Hints

```python
from typing import TypeVar, Generic, Protocol, TypedDict, Literal

# TypeVar for generics
T = TypeVar("T")
def first(items: list[T]) -> T | None:
    return items[0] if items else None

# TypedDict for structured dicts
class UserDict(TypedDict):
    name: str
    age: int
    email: str | None

# Literal for specific values
Status = Literal["pending", "approved", "rejected"]
def set_status(status: Status) -> None:
    pass

# Protocol for structural typing
class Readable(Protocol):
    def read(self) -> str: ...

def process_readable(obj: Readable) -> str:
    return obj.read()
```

### Pydantic Models (Preferred for Data Validation)

```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class SourceRequest(BaseModel):
    url: str = Field(..., description="The source URL to check")
    force_refresh: bool = Field(default=False)
    
    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

class ComplianceDecision(BaseModel):
    source_id: str
    status: Literal["ALLOWED", "CONDITIONAL", "BLOCKED", "REVIEW_REQUIRED"]
    confidence: float = Field(ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {"frozen": True}  # Immutable
```

## Async/Await Patterns

### Basic Async Functions

```python
import asyncio
import httpx

async def fetch_url(url: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text

async def fetch_multiple(urls: list[str]) -> list[str]:
    """Fetch multiple URLs concurrently."""
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for response in responses:
            if isinstance(response, Exception):
                results.append(f"Error: {response}")
            else:
                results.append(response.text)
        return results
```

### Async Context Managers

```python
from contextlib import asynccontextmanager
from typing import AsyncGenerator

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    session = AsyncSession(engine)
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

# Usage
async def create_user(user_data: dict) -> User:
    async with get_db_session() as session:
        user = User(**user_data)
        session.add(user)
        return user
```

### Async Patterns to Avoid

```python
# BAD: Blocking call in async function
async def bad_fetch():
    import requests  # Blocking!
    return requests.get("https://example.com")

# GOOD: Use async HTTP client
async def good_fetch():
    async with httpx.AsyncClient() as client:
        return await client.get("https://example.com")

# BAD: Sequential awaits when concurrent is possible
async def bad_sequential():
    result1 = await fetch_url("url1")
    result2 = await fetch_url("url2")  # Waits for result1
    return result1, result2

# GOOD: Concurrent execution
async def good_concurrent():
    result1, result2 = await asyncio.gather(
        fetch_url("url1"),
        fetch_url("url2"),
    )
    return result1, result2
```

## Error Handling

### Exception Hierarchy

```python
# Define custom exceptions
class ACCError(Exception):
    """Base exception for ACC application."""
    pass

class FetchError(ACCError):
    """Error during content fetching."""
    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"Failed to fetch {url}: {reason}")

class AnalysisError(ACCError):
    """Error during AI analysis."""
    pass

class ValidationError(ACCError):
    """Input validation error."""
    pass
```

### Exception Handling Patterns

```python
# Specific exceptions first
try:
    result = await fetch_and_analyze(url)
except httpx.TimeoutException:
    logger.warning(f"Timeout fetching {url}")
    raise FetchError(url, "timeout")
except httpx.HTTPStatusError as e:
    logger.error(f"HTTP error {e.response.status_code} for {url}")
    raise FetchError(url, f"HTTP {e.response.status_code}")
except Exception as e:
    logger.exception(f"Unexpected error for {url}")
    raise ACCError(f"Unexpected error: {e}") from e

# Context managers for cleanup
from contextlib import suppress

# Suppress specific exceptions
with suppress(FileNotFoundError):
    os.remove(temp_file)
```

## Project Structure

```
project-root/
├── src/
│   └── package_name/
│       ├── __init__.py
│       ├── config.py          # Configuration management
│       ├── models.py          # Data models (Pydantic/SQLAlchemy)
│       ├── api/               # API layer
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── routes/
│       │   └── schemas.py
│       ├── services/          # Business logic
│       │   ├── __init__.py
│       │   └── ...
│       └── db/                # Database layer
│           ├── __init__.py
│           └── ...
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures
│   ├── unit/
│   └── integration/
├── pyproject.toml            # Project config (PEP 517/518)
├── requirements.txt          # Or requirements-*.txt
├── .env.example
├── .gitignore
└── README.md
```

## Configuration Management

```python
from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr
from functools import lru_cache

class Settings(BaseSettings):
    # Database
    database_url: str = Field(..., validation_alias="DATABASE_URL")
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False
    
    # Secrets
    api_key: SecretStr = Field(..., validation_alias="API_KEY")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

## Logging Best Practices

```python
import logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger(__name__)

# Usage with context
logger.info(
    "processing_source",
    url=url,
    source_id=source_id,
    action="fetch",
)
```

## Testing Patterns

See the `testing-pytest` skill for comprehensive testing guidelines.

## Code Review Checklist

- [ ] Type hints on all public functions
- [ ] Docstrings on public modules, classes, and functions
- [ ] No hardcoded secrets or credentials
- [ ] Proper error handling with specific exceptions
- [ ] Async code uses async libraries (no blocking calls)
- [ ] Tests cover happy path and edge cases
- [ ] Logging at appropriate levels
- [ ] No unused imports or variables
- [ ] Follows project naming conventions
