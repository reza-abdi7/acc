---
name: observability
description: Logging, metrics, and tracing patterns for monitoring and debugging ACC. Use when implementing observability features.
---

# Observability

Guide for implementing logging, metrics, and tracing in Python applications following 12-factor principles.

## When to Use This Skill

- Setting up structured logging
- Adding metrics collection
- Implementing distributed tracing
- Debugging production issues

## Structured Logging

### Setup with structlog

```python
import structlog
import logging
import sys

def configure_logging(debug: bool = False):
    """Configure structured logging."""
    
    # Processors for all logs
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    
    if debug:
        # Pretty printing for development
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        # JSON for production (stdout)
        processors.append(structlog.processors.JSONRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if debug else logging.INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

# Usage
logger = structlog.get_logger()

logger.info("processing_request", url=url, source_id=source_id)
logger.error("fetch_failed", url=url, error=str(e), exc_info=True)
```

### Request Context

```python
from contextvars import ContextVar
import uuid

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

def get_request_id() -> str:
    return request_id_var.get() or str(uuid.uuid4())

# FastAPI middleware
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request_id_var.set(request_id)
    
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        path=request.url.path,
    )
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    structlog.contextvars.unbind_contextvars("request_id", "path")
    return response
```

## Log Levels

```python
# DEBUG: Detailed diagnostic information
logger.debug("cache_lookup", key=key, hit=True)

# INFO: Normal operations, business events
logger.info("compliance_check_completed", source_id=id, status="ALLOWED")

# WARNING: Unexpected but handled situations
logger.warning("retry_attempt", url=url, attempt=2, max_retries=3)

# ERROR: Failures that need attention
logger.error("analysis_failed", source_id=id, error=str(e))

# CRITICAL: System-level failures
logger.critical("database_connection_lost", error=str(e))
```

## Metrics (Optional)

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
compliance_checks_total = Counter(
    "acc_compliance_checks_total",
    "Total compliance checks",
    ["status"]
)

# Histograms
check_duration = Histogram(
    "acc_check_duration_seconds",
    "Compliance check duration",
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0]
)

# Gauges
active_jobs = Gauge(
    "acc_active_jobs",
    "Currently active jobs"
)

# Usage
compliance_checks_total.labels(status="ALLOWED").inc()

with check_duration.time():
    result = await process_check(url)
```

## Error Tracking

```python
import traceback

def log_exception(logger, message: str, exc: Exception, **context):
    """Log exception with full context."""
    logger.error(
        message,
        error_type=type(exc).__name__,
        error_message=str(exc),
        traceback=traceback.format_exc(),
        **context
    )

# Usage
try:
    result = await fetch_document(url)
except Exception as e:
    log_exception(logger, "fetch_failed", e, url=url, source_id=source_id)
    raise
```

## Best Practices

### DO:
- **Write to stdout** - let infrastructure handle routing
- **Use structured format** (JSON) in production
- **Include context** - request_id, source_id, timestamps
- **Log at appropriate levels** - don't over-log
- **Include error details** - type, message, traceback

### DON'T:
- **Log sensitive data** - passwords, API keys, PII
- **Write to files** - use stdout/stderr
- **Use print()** - use structured logger
- **Log inside tight loops** - impacts performance
