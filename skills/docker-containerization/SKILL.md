---
name: docker-containerization
description: Dockerfile best practices, docker-compose patterns, multi-stage builds, and container optimization. Use when containerizing the ACC application.
---

# Docker Containerization

Guide for containerizing Python applications with Docker, including multi-stage builds, compose patterns, and production best practices.

## When to Use This Skill

- Writing or optimizing Dockerfiles
- Configuring docker-compose
- Debugging container issues
- Optimizing image size and build time

## Dockerfile Best Practices

### Multi-Stage Build

```dockerfile
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production image
FROM python:3.11-slim AS production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.acc.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Layer Optimization

```dockerfile
# BAD: Invalidates cache on any code change
COPY . .
RUN pip install -r requirements.txt

# GOOD: Dependencies cached separately
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
```

## Docker Compose

### Development Setup

```yaml
# docker-compose.yml
x-common: &common
  build:
    context: .
    dockerfile: Dockerfile
  volumes:
    - ./src:/app/src:ro
    - ./.env:/.env:ro
  networks:
    - acc-network

services:
  app:
    <<: *common
    profiles: [dev]
    command: uvicorn src.acc.api.main:app --host 0.0.0.0 --reload
    ports:
      - "8000:8000"
    environment:
      - DEBUG=true
    depends_on:
      postgres:
        condition: service_healthy

  postgres:
    image: postgres:15-alpine
    profiles: [dev, prod]
    environment:
      POSTGRES_USER: acc
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: acc
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U acc"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - acc-network

volumes:
  postgres_data:

networks:
  acc-network:
    driver: bridge
```

### Production with Traefik

```yaml
services:
  app:
    <<: *common
    profiles: [prod]
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1'
          memory: 1G
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.acc.rule=Host(`acc.example.com`)"
      - "traefik.http.routers.acc.tls=true"
      - "traefik.http.services.acc.loadbalancer.server.port=8000"
```

## Common Commands

```bash
# Build
docker compose build
docker compose build --no-cache

# Run
docker compose --profile dev up
docker compose --profile prod up -d

# Logs
docker compose logs -f app
docker compose logs --tail=100 app

# Shell access
docker compose exec app bash
docker compose run --rm app python -c "print('test')"

# Cleanup
docker compose down
docker compose down -v  # Remove volumes
docker system prune -a  # Remove all unused
```

## Best Practices

### DO:
- **Use multi-stage builds** to reduce image size
- **Pin base image versions** (python:3.11-slim, not python:latest)
- **Run as non-root user** in production
- **Add health checks** for orchestration
- **Use .dockerignore** to exclude unnecessary files

### DON'T:
- **Store secrets in images** - use environment variables
- **Run as root** in production containers
- **Install unnecessary packages** - keep images minimal
- **Use latest tags** - pin versions for reproducibility
