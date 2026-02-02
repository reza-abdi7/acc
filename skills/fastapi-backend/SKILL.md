---
name: fastapi-backend
description: FastAPI patterns, OpenAPI specs, async endpoints, dependency injection, Pydantic schemas, and error handling. Use when building or debugging the ACC API layer.
---

# FastAPI Backend

Comprehensive guide for building robust FastAPI applications with async patterns, proper validation, and clean architecture.

## When to Use This Skill

- Creating new API endpoints
- Implementing request/response schemas
- Setting up dependency injection
- Handling errors and validation
- Configuring OpenAPI documentation

## Project Structure

```
src/acc/api/
├── __init__.py
├── main.py              # FastAPI app instance, lifespan
├── schemas.py           # Pydantic request/response models
├── dependencies.py      # Dependency injection
├── exceptions.py        # Custom exceptions and handlers
└── routes/
    ├── __init__.py
    ├── health.py        # Health check endpoints
    ├── compliance.py    # Compliance check endpoints
    └── sources.py       # Source management endpoints
```

## Application Setup

```python
# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.acc.api.routes import health, compliance, sources
from src.acc.api.exceptions import register_exception_handlers
from src.acc.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db_pool()
    yield
    # Shutdown
    await close_db_pool()

app = FastAPI(
    title="ACC API",
    description="Automated Compliance Check API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.debug else None,
    redoc_url="/api/redoc" if settings.debug else None,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router, tags=["Health"])
app.include_router(compliance.router, prefix="/api/v1", tags=["Compliance"])
app.include_router(sources.router, prefix="/api/v1", tags=["Sources"])

# Exception handlers
register_exception_handlers(app)
```

## Pydantic Schemas

```python
# schemas.py
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
from typing import Literal

# Request schemas
class ComplianceCheckRequest(BaseModel):
    url: str = Field(..., description="Source URL to check", examples=["https://example.com"])
    force_refresh: bool = Field(default=False, description="Force re-fetch even if cached")
    
    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

# Response schemas
class ComplianceDecisionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    source_id: str
    status: Literal["ALLOWED", "CONDITIONAL", "BLOCKED", "REVIEW_REQUIRED"]
    compliance_channel: str | None = None
    constraints: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    review_flag: bool = False
    created_at: datetime
    version: dict

class SourceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: str
    url: str
    domain: str
    last_checked: datetime | None
    decision_status: str | None

# Pagination
class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    page_size: int
    pages: int

# Error responses
class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
```

## Route Implementation

```python
# routes/compliance.py
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from src.acc.api.schemas import ComplianceCheckRequest, ComplianceDecisionResponse
from src.acc.api.dependencies import get_orchestrator, get_current_user
from src.acc.orchestrator import Orchestrator

router = APIRouter(prefix="/compliance")

@router.post(
    "/check",
    response_model=ComplianceDecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Request compliance check",
    responses={
        200: {"description": "Compliance decision"},
        202: {"description": "Check in progress"},
        400: {"description": "Invalid URL"},
        500: {"description": "Internal error"},
    },
)
async def check_compliance(
    request: ComplianceCheckRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ComplianceDecisionResponse:
    """
    Request a compliance check for a source URL.
    
    Returns the compliance decision with status, constraints, and confidence.
    """
    decision = await orchestrator.process_compliance_check(
        source_url=request.url,
        force_refresh=request.force_refresh,
    )
    return ComplianceDecisionResponse.model_validate(decision)

@router.get(
    "/status/{source_id}",
    response_model=ComplianceDecisionResponse,
    summary="Get compliance status",
)
async def get_compliance_status(
    source_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ComplianceDecisionResponse:
    """Get the current compliance status for a source."""
    decision = await orchestrator.get_decision(source_id)
    if not decision:
        raise HTTPException(status_code=404, detail="Source not found")
    return ComplianceDecisionResponse.model_validate(decision)
```

## Dependency Injection

```python
# dependencies.py
from functools import lru_cache
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.acc.config import Settings, get_settings
from src.acc.db import get_session
from src.acc.orchestrator import Orchestrator
from src.acc.fetcher import Fetcher
from src.acc.ai_analyzer import AIAnalyzer

@lru_cache
def get_settings_cached() -> Settings:
    return get_settings()

async def get_db_session() -> AsyncSession:
    async with get_session() as session:
        yield session

def get_fetcher(settings: Settings = Depends(get_settings_cached)) -> Fetcher:
    return Fetcher(settings)

def get_analyzer(settings: Settings = Depends(get_settings_cached)) -> AIAnalyzer:
    return AIAnalyzer(settings)

def get_orchestrator(
    fetcher: Fetcher = Depends(get_fetcher),
    analyzer: AIAnalyzer = Depends(get_analyzer),
    session: AsyncSession = Depends(get_db_session),
) -> Orchestrator:
    return Orchestrator(fetcher=fetcher, analyzer=analyzer, session=session)

# Request context
def get_request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))
```

## Exception Handling

```python
# exceptions.py
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from src.acc.api.schemas import ErrorResponse

class ACCAPIError(Exception):
    def __init__(self, message: str, status_code: int = 500, detail: str | None = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail

class NotFoundError(ACCAPIError):
    def __init__(self, resource: str, id: str):
        super().__init__(f"{resource} not found", 404, f"ID: {id}")

class ValidationError(ACCAPIError):
    def __init__(self, detail: str):
        super().__init__("Validation error", 400, detail)

def register_exception_handlers(app: FastAPI):
    @app.exception_handler(ACCAPIError)
    async def acc_error_handler(request: Request, exc: ACCAPIError):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=exc.message,
                detail=exc.detail,
                request_id=request.headers.get("X-Request-ID"),
            ).model_dump(),
        )
    
    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                request_id=request.headers.get("X-Request-ID"),
            ).model_dump(),
        )
```

## Health Checks

```python
# routes/health.py
from fastapi import APIRouter, Depends
from src.acc.api.dependencies import get_db_session

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/health/ready")
async def readiness_check(session = Depends(get_db_session)):
    try:
        await session.execute("SELECT 1")
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "database": str(e)},
        )
```

## Best Practices

### DO:
- **Use Pydantic models** for all request/response validation
- **Implement dependency injection** for testability
- **Add OpenAPI documentation** with examples and descriptions
- **Handle errors consistently** with custom exception handlers
- **Use async** for all I/O operations

### DON'T:
- **Put business logic in routes** - delegate to services
- **Return raw exceptions** - always wrap in proper responses
- **Skip validation** - use Pydantic validators
- **Hardcode configuration** - use dependency injection
