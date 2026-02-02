"""SQLAlchemy ORM models for the database."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from acc.api.schemas.responses import ComplianceStatus

# Re-export for convenience
__all__ = ["Base", "ComplianceJob"]


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class ComplianceJob(Base):
    """Represents a compliance check job."""

    __tablename__ = "compliance_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    status: Mapped[ComplianceStatus] = mapped_column(
        Enum(ComplianceStatus), default=ComplianceStatus.PENDING, nullable=False
    )
    rules_applied: Mapped[list] = mapped_column(JSONB, default=list)
    rule_results: Mapped[dict] = mapped_column(JSONB, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
