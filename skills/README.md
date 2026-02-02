# ACC Skills

This directory contains Windsurf Agent Skills for the Automated Compliance Check (ACC) project.

## What are Skills?

Skills are folders containing instructions and resources that Cascade loads dynamically to improve performance on specialized tasks. They teach Cascade how to complete specific tasks in a repeatable way.

## Available Skills

### ACC Domain Skills

| Skill | Description |
|-------|-------------|
| `acc-architecture` | System architecture, components, data flow |
| `acc-fetcher` | T&C, robots.txt, LLM.txt fetching patterns |
| `acc-ai-analysis` | AI-based legal text analysis and classification |
| `acc-orchestrator` | Workflow coordination, jobs, retries |

### Development Skills

| Skill | Description |
|-------|-------------|
| `fastapi-backend` | FastAPI patterns, OpenAPI, Pydantic |
| `database-postgres` | PostgreSQL, SQLAlchemy async, migrations |
| `docker-containerization` | Dockerfile, docker-compose patterns |
| `traefik-deployment` | Traefik reverse proxy configuration |
| `llm-integration` | LLM API patterns, prompt engineering |
| `observability` | Logging, metrics, tracing |

## How to Use

### Automatic Invocation

Cascade automatically invokes skills when your request matches a skill's description. Just describe what you want to do.

### Manual Invocation

Type `@skill-name` in Cascade to explicitly activate a skill:

```
@acc-architecture explain the data flow
@fastapi-backend create a new endpoint
@database-postgres add a new model
```

## Global Skills

Global skills (available in all projects) should be copied to:

```
~/.codeium/windsurf/skills/
```

See `global-skills/` folder in project root for skills to copy:
- `ai-research-tutor` - Research methodology
- `python-best-practices` - Python coding standards
- `twelve-factor-app` - Cloud-native app methodology
- `testing-pytest` - pytest testing patterns

## Adding New Skills

1. Create a folder: `.windsurf/skills/<skill-name>/`
2. Add `SKILL.md` with YAML frontmatter:

```markdown
---
name: skill-name
description: Clear description of what this skill does and when to use it.
---

# Skill Title

Instructions and content...
```

## References

- [Windsurf Skills Documentation](https://docs.windsurf.com/windsurf/cascade/skills)
- [Agent Skills Specification](https://agentskills.io/)
