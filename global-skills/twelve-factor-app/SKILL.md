---
name: twelve-factor-app
description: The Twelve-Factor App methodology for building cloud-native, scalable, and maintainable applications. Use when designing architecture, configuring deployments, or reviewing application structure.
---

# Twelve-Factor App Methodology

A methodology for building software-as-a-service applications that are portable, scalable, and maintainable. Originally published by Heroku, these principles are essential for modern cloud-native development.

## When to Use This Skill

- Designing new application architecture
- Reviewing application structure/architecture
- Reviewing deployment configurations
- Debugging environment-related issues
- Setting up CI/CD pipelines
- Containerizing applications

## The Twelve Factors

### I. Codebase

**One codebase tracked in revision control, many deploys**

```
✓ Single repository per application
✓ Multiple deploys (dev, staging, prod) from same codebase
✓ Shared code extracted into libraries
✗ Multiple apps sharing the same codebase
```

**Implementation:**
```
# Good: One repo, multiple environments
automated-compliance-check/
├── src/
├── docker-compose.yml      # Profiles for dev/prod
├── .env.example            # Template, not actual secrets
└── Dockerfile

# Deploy different environments
docker compose --profile dev up
docker compose --profile prod up
```

### II. Dependencies

**Explicitly declare and isolate dependencies**

```
✓ All dependencies declared in manifest (pyproject.toml, requirements.txt)
✓ Dependency isolation (virtualenv, containers)
✓ No reliance on system-wide packages
✗ Implicit dependencies or system tools assumed
```

**Implementation:**
```toml
# pyproject.toml - explicit dependencies with versions
[project]
dependencies = [
    "fastapi>=0.115.0,<1.0.0",
    "httpx>=0.28.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
test = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]
```

```dockerfile
# Dockerfile - isolated environment
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
```

### III. Config

**Store config in the environment**

```
✓ Config varies between deploys (credentials, resource handles)
✓ Strict separation of config from code
✓ Environment variables for configuration
✗ Config files committed to repo
✗ Hardcoded values that change between environments
```

**Implementation:**
```python
# config.py - using pydantic-settings
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str
    api_key: str
    debug: bool = False
    log_level: str = "INFO"
    
    model_config = {"env_file": ".env"}

settings = Settings()
```

```bash
# .env.example (committed) - template only
DATABASE_URL=postgresql://user:pass@localhost:5432/db
REDIS_URL=redis://localhost:6379
API_KEY=your-api-key-here
DEBUG=false

# .env (NOT committed) - actual values
# Added to .gitignore
```

### IV. Backing Services

**Treat backing services as attached resources**

```
✓ Database, cache, queue are attached resources
✓ Swappable via configuration (URL change)
✓ No distinction between local and third-party services
✗ Hardcoded connection details
✗ Different code paths for local vs. production services
```

**Implementation:**
```python
# Same code works with local or cloud database
# Only the URL changes via environment variable

# Local development
DATABASE_URL=postgresql://localhost:5432/acc_dev

# Production (cloud)
DATABASE_URL=postgresql://user:pass@cloud-db.example.com:5432/acc_prod

# Code doesn't change
engine = create_async_engine(settings.database_url)
```

### V. Build, Release, Run

**Strictly separate build and run stages**

```
Build Stage:   Code + Dependencies → Build artifact (Docker image)
Release Stage: Build + Config → Release (tagged, immutable)
Run Stage:     Execute release in environment

✓ Immutable releases
✓ Every release has unique ID (git SHA, timestamp)
✓ Rollback by deploying previous release
✗ Modifying code at runtime
✗ Releases without version tracking
```

**Implementation:**
```dockerfile
# Build stage
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/

# Release: tagged image
# docker build -t acc:v1.2.3 .
# docker build -t acc:$(git rev-parse --short HEAD) .
```

```yaml
# docker-compose.yml
services:
  app:
    image: acc:${VERSION:-latest}  # Release version from env
    environment:
      - DATABASE_URL=${DATABASE_URL}  # Config injected at run
```

### VI. Processes

**Execute the app as one or more stateless processes**

```
✓ Processes are stateless and share-nothing
✓ Persistent data in backing services (database, object storage)
✓ Sticky sessions stored in datastore (Redis)
✗ In-memory state that survives restarts
✗ Local filesystem for persistent storage
```

**Implementation:**
```python
# BAD: In-memory state
class BadService:
    cache = {}  # Lost on restart!
    
    def get_data(self, key):
        if key not in self.cache:
            self.cache[key] = self.fetch(key)
        return self.cache[key]

# GOOD: External state store
class GoodService:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get_data(self, key):
        cached = await self.redis.get(key)
        if cached is None:
            data = await self.fetch(key)
            await self.redis.set(key, data, ex=3600)
            return data
        return cached
```

### VII. Port Binding

**Export services via port binding**

```
✓ App is self-contained
✓ Exports HTTP (or other protocol) by binding to port
✓ No runtime injection into web server
✗ Relying on external web server injection
```

**Implementation:**
```python
# main.py - self-contained HTTP server
import uvicorn
from fastapi import FastAPI

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )
```

```dockerfile
EXPOSE 8000
CMD ["uvicorn", "src.acc.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### VIII. Concurrency

**Scale out via the process model**

```
✓ Different process types for different workloads
✓ Horizontal scaling by adding processes
✓ Process manager handles distribution
✗ Threading for scaling (use processes)
✗ Daemonization inside the app
```

**Implementation:**
```yaml
# docker-compose.yml - different process types
services:
  web:
    image: acc:latest
    command: uvicorn src.acc.api.main:app --host 0.0.0.0
    deploy:
      replicas: 3  # Scale web processes
  
  worker:
    image: acc:latest
    command: python -m src.acc.worker
    deploy:
      replicas: 2  # Scale workers independently
```

### IX. Disposability

**Maximize robustness with fast startup and graceful shutdown**

```
✓ Fast startup (seconds, not minutes)
✓ Graceful shutdown on SIGTERM
✓ Robust against sudden death
✓ Idempotent operations (safe to retry)
✗ Long startup initialization
✗ Jobs that can't be interrupted
```

**Implementation:**
```python
import signal
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db_pool()
    yield
    # Shutdown - graceful cleanup
    await close_db_pool()
    await cancel_pending_tasks()

app = FastAPI(lifespan=lifespan)

# Graceful shutdown handler
async def shutdown():
    logger.info("Shutting down gracefully...")
    # Complete in-flight requests
    # Close connections
    # Flush logs
```

### X. Dev/Prod Parity

**Keep development, staging, and production as similar as possible**

```
✓ Same backing services in all environments
✓ Same deployment process
✓ Continuous deployment (small time gap)
✗ Different databases (SQLite dev, Postgres prod)
✗ Different OS or runtime versions
```

**Implementation:**
```yaml
# docker-compose.yml - same services, different configs
x-service-common: &service-common
  image: acc:latest
  depends_on:
    - postgres
    - redis

services:
  app_dev:
    <<: *service-common
    profiles: [dev]
    environment:
      - DEBUG=true
  
  app_prod:
    <<: *service-common
    profiles: [prod]
    environment:
      - DEBUG=false
```

### XI. Logs

**Treat logs as event streams**

```
✓ Write to stdout/stderr
✓ No log file management in app
✓ Structured logging (JSON)
✓ Execution environment handles routing
✗ Writing to log files
✗ Log rotation in application
```

**Implementation:**
```python
import structlog
import sys

# Configure to write to stdout
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
)

logger = structlog.get_logger()

# Usage - structured events
logger.info("request_processed", 
    url=url, 
    status="success",
    duration_ms=150
)
```

```yaml
# Docker captures stdout
docker logs container_name

# Or route to log aggregator
logging:
  driver: "json-file"
  options:
    max-size: "10m"
```

### XII. Admin Processes

**Run admin/management tasks as one-off processes**

```
✓ Run in identical environment as app
✓ Same codebase and config
✓ Database migrations as one-off process
✗ SSH into server to run scripts
✗ Different code for admin tasks
```

**Implementation:**
```bash
# Run migration in same environment
docker compose run --rm app alembic upgrade head

# Run one-off task
docker compose run --rm app python -m src.acc.scripts.backfill

# Interactive shell
docker compose run --rm app python
```

```python
# scripts/backfill.py - uses same config and models
from src.acc.config import settings
from src.acc.db import get_session
from src.acc.models import Source

async def main():
    async with get_session() as session:
        # Admin task using production code
        pass

if __name__ == "__main__":
    asyncio.run(main())
```

## Quick Reference Checklist

| Factor | Question | Check |
|--------|----------|-------|
| Codebase | Single repo, multiple deploys? | ☐ |
| Dependencies | All deps in manifest, isolated? | ☐ |
| Config | All config from environment? | ☐ |
| Backing Services | Services swappable via URL? | ☐ |
| Build/Release/Run | Stages strictly separated? | ☐ |
| Processes | Stateless, share-nothing? | ☐ |
| Port Binding | Self-contained, exports port? | ☐ |
| Concurrency | Scales via process model? | ☐ |
| Disposability | Fast start, graceful stop? | ☐ |
| Dev/Prod Parity | Environments similar? | ☐ |
| Logs | Writes to stdout, structured? | ☐ |
| Admin Processes | One-off in same environment? | ☐ |

## References

- [12factor.net](https://12factor.net/) - Original methodology
- [Beyond the Twelve-Factor App](https://www.oreilly.com/library/view/beyond-the-twelve-factor/9781492042631/) - Modern extensions
