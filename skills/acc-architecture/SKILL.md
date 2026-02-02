---
name: acc-architecture
description: Automated Compliance Check (ACC) system architecture, component responsibilities, data flow, and decision model. Use when understanding the system, implementing features, or making architectural decisions.
---

# ACC Architecture

The Automated Compliance Check (ACC) system automates compliance assessment of external web sources by analyzing their Terms & Conditions (T&C), robots.txt, and LLM.txt files against pre-defined legal rules.

## When to Use This Skill

- Understanding how ACC components interact
- Implementing new features
- Debugging data flow issues
- Making architectural decisions
- Onboarding to the project

## System Overview

ACC is an internal tool that:
- Analyzes T&C, robots.txt, and LLM.txt from external sources
- Produces compliance decisions (ALLOWED, CONDITIONAL, BLOCKED, REVIEW_REQUIRED)
- Provides auditable, reproducible, and explainable decisions
- Supports human-in-the-loop review for uncertain cases

## Core Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Client Input (Source URL)                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           API Endpoints                              │
│                    (Request/Response Interface)                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     App/Orchestrator (Workflow)                      │
│           (Jobs, Queues, Idempotency, Retries, Timeouts)            │
└─────────────────────────────────────────────────────────────────────┘
                    │                               │
                    ▼                               ▼
┌───────────────────────────┐       ┌───────────────────────────────┐
│         Fetcher           │       │        AI Analysis            │
│  (T&C, robots.txt, LLM.txt)│       │  (Classification, Decisions)  │
└───────────────────────────┘       └───────────────────────────────┘
                    │                               │
                    ▼                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DB (Postgres / Data Lake)                         │
│                      (Object Storage)                                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Rules and Logics from Compliance Manager                │
│                    (Pre-defined, Updateable)                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Observability                                │
│                     (Logs, Metrics, Tracing)                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### 1. API Endpoints (`src/acc/api/`)

**Purpose**: External interface for ACC system

**Responsibilities**:
- Receive compliance check requests (Source URL)
- Return compliance decisions
- Provide query endpoints for existing decisions
- Handle authentication and rate limiting

**Key Principle**: Minimal input surface (just URL), single authoritative output (decision)

```python
# Input: Source URL only
POST /api/v1/compliance/check
{
    "url": "https://example.com"
}

# Output: Compliance Decision
{
    "status": "ALLOWED" | "CONDITIONAL" | "BLOCKED" | "REVIEW_REQUIRED",
    "compliance_channel": "...",
    "constraints": [...],
    "confidence": 0.95,
    "version": {...}
}
```

### 2. Orchestrator (`src/acc/orchestrator/`)

**Purpose**: Workflow coordination and job management

**Responsibilities**:
- Control workflow: fetch → parse → analyze → decision
- Manage job queues and scheduling
- Handle retries and timeouts
- Ensure idempotency
- Trigger Fetcher and AI Analyzer

**Workflow**:
```
Request → Orchestrator → Fetcher → DB → AI Analyzer → DB → Decision
```

### 3. Fetcher (`src/acc/fetcher/`)

**Purpose**: Retrieve legal documents from external sources

**Responsibilities**:
- Fetch T&C pages from websites
- Fetch robots.txt files
- Fetch LLM.txt files (if available)
- Normalize URLs and resolve domains
- Store raw documents in DB with metadata (language, geo, timestamp)
- Handle various content types and encodings

**Documents Fetched**:
| Document | Purpose |
|----------|---------|
| T&C | Terms and Conditions - primary legal text |
| robots.txt | Web crawler permissions |
| LLM.txt | AI/LLM-specific usage permissions |

### 4. AI Analyzer (`src/acc/ai_analyzer/`)

**Purpose**: Analyze legal text and produce compliance decisions

**Responsibilities**:
- Query DB for fetched documents
- Extract relevant clauses from legal text
- Classify relevance (relevant/irrelevant)
- Detect permissions, prohibitions, conditions
- Apply compliance rules (from Compliance Manager)
- Label and classify T&Cs
- Produce compliance decision with confidence score

**Output Signals** (internal):
- Commercial use allowed/prohibited
- Attribution requirements
- AI training permissions
- Data processing restrictions
- Geographic restrictions

### 5. Rules Engine (Compliance Manager)

**Purpose**: Define compliance logic and classification rules

**Characteristics**:
- Pre-defined by Compliance Manager
- May be updated over time
- Used by AI Analyzer for classification
- Stored separately from code (configurable)

### 6. Database (`src/acc/db/`)

**Purpose**: Persistent storage for all ACC data

**Stores**:
- Source metadata (URL, domain, family)
- Raw fetched documents (T&C, robots.txt, LLM.txt)
- Document metadata (language, geo, fetch timestamp)
- AI analysis results and signals
- Compliance decisions (versioned)
- Audit trail

### 7. Observability

**Purpose**: Monitoring, logging, and tracing

**Emits**:
- Structured logs (JSON to stdout)
- Metrics (processing time, success/failure rates)
- Traces (request flow through components)

## Data Flow

### Compliance Check Request Flow

```
1. Client sends URL to API
2. API creates job, sends to Orchestrator
3. Orchestrator checks if recent decision exists
   - If yes and fresh: return cached decision
   - If no or stale: continue
4. Orchestrator triggers Fetcher
5. Fetcher:
   - Resolves URL to domain
   - Fetches T&C, robots.txt, LLM.txt
   - Stores raw documents in DB
6. Orchestrator triggers AI Analyzer
7. AI Analyzer:
   - Retrieves documents from DB
   - Analyzes legal text
   - Applies compliance rules
   - Produces decision
   - Stores results in DB
8. Orchestrator returns decision to API
9. API returns decision to Client
```

### Re-check Flow (UC-02)

```
1. Scheduled job or manual request triggers re-check
2. Orchestrator fetches fresh documents
3. AI Analyzer compares with previous analysis
4. If changed: new decision created
5. If unchanged: existing decision confirmed
6. Version history maintained
```

## Decision Model

### Decision Statuses

| Status | Meaning |
|--------|---------|
| `ALLOWED` | Source can be used without restrictions |
| `CONDITIONAL` | Source can be used with specific constraints |
| `BLOCKED` | Source must not be used |
| `REVIEW_REQUIRED` | Human review needed (low confidence or edge case) |

### Decision Object

```python
class ComplianceDecision:
    source_id: str
    status: Literal["ALLOWED", "CONDITIONAL", "BLOCKED", "REVIEW_REQUIRED"]
    compliance_channel: str
    constraints: list[Constraint]  # What is allowed/forbidden
    confidence: float  # 0.0 to 1.0
    review_flag: bool
    version: DecisionVersion
    created_at: datetime
    
class DecisionVersion:
    model_version: str
    rules_version: str
    timestamp: datetime
```

## Design Principles

### 1. Minimal Input Surface
- Only URL required as input
- No user-provided legal interpretation
- Avoids configuration drift and bias

### 2. Single Authoritative Output
- One decision per source
- Everything else is internal by-product
- Stored for auditability, not exposed as API

### 3. Reproducibility
- Same input → same output (given same rules/model)
- All decisions versioned
- Audit trail maintained

### 4. Human-in-the-Loop
- Low confidence → REVIEW_REQUIRED
- Humans can confirm or override
- Overrides recorded in audit trail

### 5. Separation of Concerns
- Fetcher: only fetches, doesn't analyze
- AI Analyzer: only analyzes, doesn't fetch
- Orchestrator: coordinates, doesn't do work
- API: interface only, no business logic

## Directory Structure

```
src/acc/
├── __init__.py
├── config.py              # Configuration management
├── models.py              # Shared data models
├── api/                   # API layer
│   ├── main.py
│   ├── routes/
│   └── schemas.py
├── orchestrator/          # Workflow coordination
│   └── __init__.py
├── fetcher/               # Document fetching
│   └── fetcher.py
├── ai_analyzer/           # AI analysis
│   └── __init__.py
├── db/                    # Database layer
│   └── ...
├── helpers/               # Shared utilities
│   └── ...
└── infra/                 # Infrastructure utilities
    └── ...
```

## What ACC Is NOT

- **Not a legal advisor**: Supports decisions, doesn't make legal judgments
- **Not a contract enforcer**: Provides guidance, doesn't enforce
- **Not a replacement for lawyers**: Assists, doesn't replace human expertise
- **Not a business strategy tool**: Compliance only, not business decisions

## References

- `@.ai/use-case.md` - Detailed use cases
- `@.ai/rules.md` - Compliance rules documentation
- `@.ai/design-guideline.md` - Design guidelines
