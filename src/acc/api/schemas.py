from datetime import datetime
from enum import Enum
from typing import Dict, List
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class ComplianceCheckRequest(BaseModel):
    """Request payload for checking compliance of a source.

    Attributes:
        source_url: The URL of the source to check for compliance.
    """

    source_url: HttpUrl = Field(..., description="URL of the source to check")


class ComplianceStatus(str, Enum):
    """Status of the compliance check."""

    FULL_OPEN = "Full_Open"
    PARTIAL_OPEN = "Partial_Open"
    OPEN_RISK_1 = "Open_Risk_1"
    OPEN_RISK_2 = "Open_Risk_2"
    OPEN_RISK_3 = "Open_Risk_3"
    BLOCKED = "Blocked"
    APPROVED = "APPROVED"
    BLOCKED = "BLOCKED"
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    ERROR = "ERROR"


class ComplianceCheckResponse(BaseModel):
    """Response model for compliance check results.

    Attributes:
        job_id: Unique identifier for the compliance check job.
        source_url: URL that was checked.
        status: The compliance status result.
        rules_applied: List of rule names that were applied.
        rule_results: Mapping of rule name to pass/fail result.
        error: Error message if status is ERROR.
        created_at: Timestamp when the job was created.
        updated_at: Timestamp when the job was last updated.
    """

    job_id: UUID
    source_url: HttpUrl
    status: ComplianceStatus
    rules_applied: List[str] = Field(default_factory=list)
    rule_results: Dict[str, bool] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime
    updated_at: datetime