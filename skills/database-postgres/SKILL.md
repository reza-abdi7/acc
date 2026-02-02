---
name: database-postgres
description: PostgreSQL patterns with SQLAlchemy async, migrations, connection pooling, and query optimization. Use when working with the ACC database layer.
---

# Database PostgreSQL

Guide for working with PostgreSQL using SQLAlchemy async ORM, including models, migrations, and best practices.

## When to Use This Skill

- Defining database models
- Writing async database queries
- Setting up connection pooling
- Creating and running migrations
- Optimizing query performance

## Project Structure

```
src/acc/db/
├── __init__.py          # Database initialization, session factory
├── models.py            # SQLAlchemy ORM models
├── repositories.py      # Data access layer
└── migrations/          # Alembic migrations
    ├── env.py
    ├── alembic.ini
    └── versions/
```

## Database Setup

```python
# db/__init__.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from contextlib import asynccontextmanager

from src.acc.config import settings

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections after 1 hour
)

# Session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

@asynccontextmanager
async def get_session() -> AsyncSession:
    """Get database session with automatic cleanup."""
    session = async_session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

async def init_db():
    """Initialize database (create tables)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def close_db():
    """Close database connections."""
    await engine.dispose()
```

## SQLAlchemy Models

```python
# db/models.py
from sqlalchemy import String, Text, Float, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from typing import Optional

class Base(DeclarativeBase):
    pass

class Source(Base):
    __tablename__ = "sources"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    normalized_url: Mapped[str] = mapped_column(String(2048), nullable=False, unique=True)
    domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    documents: Mapped[list["Document"]] = relationship(back_populates="source", cascade="all, delete-orphan")
    decisions: Mapped[list["ComplianceDecision"]] = relationship(back_populates="source", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("ix_sources_domain_created", "domain", "created_at"),
    )

class Document(Base):
    __tablename__ = "documents"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id"), nullable=False, index=True)
    document_type: Mapped[str] = mapped_column(String(20), nullable=False)  # tc, robots, llm_txt
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
    extracted_text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(10))
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    version: Mapped[int] = mapped_column(default=1)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    source: Mapped["Source"] = relationship(back_populates="documents")
    
    __table_args__ = (
        Index("ix_documents_source_type", "source_id", "document_type"),
    )

class ComplianceDecision(Base):
    __tablename__ = "compliance_decisions"
    
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    compliance_channel: Mapped[Optional[str]] = mapped_column(String(50))
    constraints: Mapped[list] = mapped_column(JSONB, default=list)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    review_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    signals: Mapped[dict] = mapped_column(JSONB, default=dict)
    version_info: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    source: Mapped["Source"] = relationship(back_populates="decisions")
```

## Repository Pattern

```python
# db/repositories.py
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

class SourceRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, source_id: str) -> Optional[Source]:
        result = await self.session.execute(
            select(Source).where(Source.id == source_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_url(self, normalized_url: str) -> Optional[Source]:
        result = await self.session.execute(
            select(Source).where(Source.normalized_url == normalized_url)
        )
        return result.scalar_one_or_none()
    
    async def create(self, source: Source) -> Source:
        self.session.add(source)
        await self.session.flush()
        return source
    
    async def list_by_domain(
        self,
        domain: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Source]:
        result = await self.session.execute(
            select(Source)
            .where(Source.domain == domain)
            .order_by(Source.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

class DecisionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_latest(self, source_id: str) -> Optional[ComplianceDecision]:
        result = await self.session.execute(
            select(ComplianceDecision)
            .where(ComplianceDecision.source_id == source_id)
            .order_by(ComplianceDecision.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def create(self, decision: ComplianceDecision) -> ComplianceDecision:
        self.session.add(decision)
        await self.session.flush()
        return decision
```

## Migrations with Alembic

```python
# migrations/env.py
from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine
from src.acc.db.models import Base
from src.acc.config import settings

target_metadata = Base.metadata

def run_migrations_offline():
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
    )
    with context.begin_transaction():
        context.run_migrations()

async def run_migrations_online():
    engine = create_async_engine(settings.database_url)
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await engine.dispose()

def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()
```

```bash
# Migration commands
alembic revision --autogenerate -m "Add sources table"
alembic upgrade head
alembic downgrade -1
alembic history
```

## Query Patterns

```python
# Eager loading relationships
async def get_source_with_documents(session: AsyncSession, source_id: str):
    result = await session.execute(
        select(Source)
        .options(selectinload(Source.documents))
        .where(Source.id == source_id)
    )
    return result.scalar_one_or_none()

# Aggregation
async def count_by_status(session: AsyncSession) -> dict[str, int]:
    result = await session.execute(
        select(
            ComplianceDecision.status,
            func.count(ComplianceDecision.id)
        )
        .group_by(ComplianceDecision.status)
    )
    return dict(result.all())

# Bulk insert
async def bulk_create_documents(session: AsyncSession, documents: list[Document]):
    session.add_all(documents)
    await session.flush()
```

## Best Practices

### DO:
- **Use async sessions** for all database operations
- **Define indexes** for frequently queried columns
- **Use repository pattern** to abstract data access
- **Run migrations** for schema changes
- **Use connection pooling** with appropriate limits

### DON'T:
- **Commit in repository methods** - let the caller control transactions
- **Use raw SQL** unless necessary for performance
- **Forget to close sessions** - use context managers
- **Skip migrations** - never modify production schema manually
