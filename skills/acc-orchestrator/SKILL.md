---
name: acc-orchestrator
description: Workflow orchestration, job management, idempotency, retries, and timeouts for ACC. Use when implementing or debugging the Orchestrator component.
---

# ACC Orchestrator

The Orchestrator coordinates the compliance check workflow, managing fetch → analyze → decision pipeline with reliability guarantees.

## When to Use This Skill

- Implementing workflow coordination
- Designing job queues and scheduling
- Handling retries and failure recovery
- Ensuring idempotency
- Debugging workflow issues

## Workflow Overview

```
Request → Validation → Cache Check → [Fetch Docs] → [AI Analysis] → Decision
                           ↓
                    (Cache Hit → Return)
```

### Job States

```
PENDING → RUNNING → COMPLETED
              ↓
           FAILED → RETRYING → (back to RUNNING)
              ↓
           TIMEOUT
```

## Data Models

```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"

class JobStep(str, Enum):
    VALIDATION = "validation"
    FETCH_TC = "fetch_tc"
    FETCH_ROBOTS = "fetch_robots"
    FETCH_LLM_TXT = "fetch_llm_txt"
    ANALYSIS = "analysis"
    DECISION = "decision"

class Job(BaseModel):
    id: str
    source_url: str
    normalized_url: str
    idempotency_key: str
    status: JobStatus = JobStatus.PENDING
    current_step: JobStep | None = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    timeout_at: datetime | None = None
    error: str | None = None
    result: dict | None = None
```

## Idempotency

```python
import hashlib
from datetime import datetime, timedelta

def generate_idempotency_key(url: str, window_hours: int = 24) -> str:
    """Generate idempotency key for URL within time window."""
    normalized = normalize_url(url)
    window_start = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    window_start -= timedelta(hours=window_start.hour % window_hours)
    
    key_input = f"{normalized}:{window_start.isoformat()}"
    return hashlib.sha256(key_input.encode()).hexdigest()[:16]

async def create_or_get_job(session: AsyncSession, url: str) -> tuple[Job, bool]:
    """Create new job or return existing (idempotent)."""
    normalized = normalize_url(url)
    key = generate_idempotency_key(normalized)
    
    existing = await session.execute(
        select(Job).where(
            Job.idempotency_key == key,
            Job.status.in_([JobStatus.COMPLETED, JobStatus.RUNNING, JobStatus.PENDING])
        )
    )
    if job := existing.scalar_one_or_none():
        return job, False
    
    job = Job(
        id=generate_job_id(),
        source_url=url,
        normalized_url=normalized,
        idempotency_key=key,
        timeout_at=datetime.utcnow() + timedelta(minutes=30),
    )
    session.add(job)
    await session.commit()
    return job, True
```

## Workflow Execution

```python
class Orchestrator:
    def __init__(self, fetcher: Fetcher, analyzer: AIAnalyzer, db: Database):
        self.fetcher = fetcher
        self.analyzer = analyzer
        self.db = db
    
    async def process_compliance_check(self, source_url: str) -> ComplianceDecision:
        async with self.db.session() as session:
            job, is_new = await create_or_get_job(session, source_url)
            
            if not is_new and job.status == JobStatus.COMPLETED:
                return ComplianceDecision(**job.result)
            
            if not is_new and job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
                return await self._wait_for_job(job.id)
            
            try:
                return await self._execute_workflow(session, job)
            except Exception as e:
                await self._handle_failure(session, job, e)
                raise
    
    async def _execute_workflow(self, session: AsyncSession, job: Job) -> ComplianceDecision:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        await session.commit()
        
        # Fetch documents
        tc_doc = await self._step(job, JobStep.FETCH_TC,
            self.fetcher.fetch_tc(job.normalized_url))
        
        robots = await self._step(job, JobStep.FETCH_ROBOTS,
            self.fetcher.fetch_robots(job.normalized_url), optional=True)
        
        llm_txt = await self._step(job, JobStep.FETCH_LLM_TXT,
            self.fetcher.fetch_llm_txt(job.normalized_url), optional=True)
        
        # Analyze
        signals = await self._step(job, JobStep.ANALYSIS,
            self.analyzer.analyze(tc_doc, robots, llm_txt))
        
        # Decision
        decision = await self._step(job, JobStep.DECISION,
            self.analyzer.synthesize_decision(job.id, signals))
        
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.result = decision.model_dump()
        await session.commit()
        
        return decision
```

## Retry Logic

```python
from dataclasses import dataclass

@dataclass
class RetryConfig:
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0

def is_retryable(error: Exception) -> bool:
    """Determine if error is retryable."""
    if isinstance(error, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in [429, 502, 503, 504]
    if isinstance(error, FetchError):
        return error.retryable
    return False

async def execute_with_retry(
    action: Callable,
    config: RetryConfig = RetryConfig(),
) -> Any:
    """Execute action with exponential backoff retry."""
    last_error = None
    
    for attempt in range(config.max_retries + 1):
        try:
            return await action()
        except Exception as e:
            last_error = e
            if not is_retryable(e) or attempt == config.max_retries:
                raise
            
            delay = min(
                config.initial_delay * (config.exponential_base ** attempt),
                config.max_delay
            )
            await asyncio.sleep(delay)
    
    raise last_error
```

## Timeout Management

```python
async def _step(
    self,
    job: Job,
    step: JobStep,
    coro: Coroutine,
    optional: bool = False,
    timeout: int = 300,
) -> Any | None:
    """Execute step with timeout."""
    job.current_step = step
    
    if datetime.utcnow() > job.timeout_at:
        raise JobTimeoutError(job.id)
    
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        if optional:
            logger.warning(f"Optional step {step} timed out")
            return None
        raise StepTimeoutError(job.id, step)
    except Exception as e:
        if optional and isinstance(e, (FetchError, DocumentNotFoundError)):
            return None
        raise
```

## Error Handling

```python
class OrchestratorError(Exception):
    """Base orchestrator error."""
    pass

class JobTimeoutError(OrchestratorError):
    def __init__(self, job_id: str):
        super().__init__(f"Job {job_id} timed out")

class StepTimeoutError(OrchestratorError):
    def __init__(self, job_id: str, step: JobStep):
        super().__init__(f"Step {step} timed out for job {job_id}")

class StepFailedError(OrchestratorError):
    def __init__(self, step: JobStep, reason: str):
        super().__init__(f"Step {step} failed: {reason}")

async def _handle_failure(self, session: AsyncSession, job: Job, error: Exception):
    """Handle job failure."""
    if is_retryable(error) and job.retry_count < job.max_retries:
        job.status = JobStatus.RETRYING
        job.retry_count += 1
        job.error = str(error)
    else:
        job.status = JobStatus.FAILED
        job.error = str(error)
        job.completed_at = datetime.utcnow()
    
    await session.commit()
    logger.error(f"Job {job.id} failed: {error}", exc_info=True)
```

## Best Practices

### DO:
- **Generate idempotency keys** from normalized URL + time window
- **Use exponential backoff** for retries
- **Set timeouts** at both job and step level
- **Log step transitions** for debugging
- **Store job results** for cache hits

### DON'T:
- **Retry non-retryable errors** (validation, 404)
- **Block indefinitely** - always have timeouts
- **Lose job state** - persist before long operations
- **Ignore partial failures** - handle optional steps gracefully
