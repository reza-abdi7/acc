---
name: traefik-deployment
description: Traefik reverse proxy configuration, TLS setup, routing rules, and Docker labels. Use when configuring ACC deployment with Traefik.
---

# Traefik Deployment

Guide for deploying applications behind Traefik reverse proxy with TLS, routing, and load balancing.

## When to Use This Skill

- Configuring Traefik routing for ACC
- Setting up TLS/HTTPS
- Debugging routing issues
- Load balancing multiple instances

## Traefik with Docker Labels

### Basic Service Configuration

```yaml
# docker-compose.yml
services:
  app:
    image: acc:latest
    labels:
      # Enable Traefik
      - "traefik.enable=true"
      
      # Router configuration
      - "traefik.http.routers.acc.rule=Host(`acc.example.com`)"
      - "traefik.http.routers.acc.entrypoints=websecure"
      - "traefik.http.routers.acc.tls=true"
      
      # Service configuration
      - "traefik.http.services.acc.loadbalancer.server.port=8000"
    networks:
      - traefik_network

networks:
  traefik_network:
    external: true
```

### Path-Based Routing

```yaml
labels:
  # Route with path prefix
  - "traefik.http.routers.acc.rule=Host(`worker.example.com`) && PathPrefix(`/acc`)"
  
  # Strip prefix before forwarding
  - "traefik.http.middlewares.acc-strip.stripprefix.prefixes=/acc"
  - "traefik.http.routers.acc.middlewares=acc-strip"
```

### Multiple Environments

```yaml
# Development
app_dev:
  profiles: [dev]
  labels:
    - "traefik.http.routers.acc-dev.rule=Host(`dev.example.com`) && PathPrefix(`/acc`)"
    - "traefik.http.routers.acc-dev.entrypoints=websecure"
    - "traefik.http.routers.acc-dev.tls=true"

# Production
app_prod:
  profiles: [prod]
  labels:
    - "traefik.http.routers.acc-prod.rule=Host(`prod.example.com`) && PathPrefix(`/acc`)"
    - "traefik.http.routers.acc-prod.entrypoints=websecure"
    - "traefik.http.routers.acc-prod.tls=true"
```

## Routing Rules

```yaml
# Host matching
- "traefik.http.routers.app.rule=Host(`example.com`)"

# Path matching
- "traefik.http.routers.app.rule=PathPrefix(`/api`)"

# Combined
- "traefik.http.routers.app.rule=Host(`example.com`) && PathPrefix(`/api`)"

# Regex path
- "traefik.http.routers.app.rule=PathRegexp(`^/api/v[0-9]+/.*`)"

# Headers
- "traefik.http.routers.app.rule=Headers(`X-Custom`, `value`)"
```

## Middlewares

```yaml
labels:
  # Rate limiting
  - "traefik.http.middlewares.acc-ratelimit.ratelimit.average=100"
  - "traefik.http.middlewares.acc-ratelimit.ratelimit.burst=50"
  
  # Basic auth
  - "traefik.http.middlewares.acc-auth.basicauth.users=user:$$hashed$$password"
  
  # Headers
  - "traefik.http.middlewares.acc-headers.headers.customrequestheaders.X-Forwarded-Proto=https"
  
  # Apply middlewares
  - "traefik.http.routers.acc.middlewares=acc-ratelimit,acc-headers"
```

## Health Checks

```yaml
labels:
  - "traefik.http.services.acc.loadbalancer.healthcheck.path=/health"
  - "traefik.http.services.acc.loadbalancer.healthcheck.interval=10s"
  - "traefik.http.services.acc.loadbalancer.healthcheck.timeout=3s"
```

## Debugging

```bash
# Check Traefik dashboard (if enabled)
# http://traefik.example.com/dashboard/

# View Traefik logs
docker logs traefik

# Test routing
curl -H "Host: acc.example.com" http://localhost/health

# Check container labels
docker inspect app | jq '.[0].Config.Labels'
```

## Best Practices

### DO:
- **Use TLS** for all production traffic
- **Set health checks** for proper load balancing
- **Use meaningful router names** (acc-prod, acc-dev)
- **Configure rate limiting** to prevent abuse

### DON'T:
- **Expose Traefik dashboard** without authentication
- **Use HTTP** in production
- **Forget to join Traefik network**
