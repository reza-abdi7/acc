from uuid import UUID

from acc.api.schemas.requests import ComplianceCheckRequest
from acc.api.schemas.responses import ComplianceCheckResponse
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from acc.db.models import ComplianceJob
from acc.db.session import get_db

router = APIRouter()


@router.post("/checks", response_model=ComplianceCheckResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_compliance_check(
    request: ComplianceCheckRequest,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> ComplianceCheckResponse:
    """Create a new compliance check job.

    Args:
        request: The compliance check request containing the source URL.
        db: Database session.

    Returns:
        ComplianceCheckResponse with job_id and PENDING status.
    """
    job = ComplianceJob(source_url=str(request.source_url))
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # TODO: Queue background task for actual compliance checking

    return ComplianceCheckResponse(
        job_id=job.id,
        source_url=job.source_url,
        status=job.status,
        rules_applied=job.rules_applied,
        rule_results=job.rule_results,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get("/checks/{job_id}", response_model=ComplianceCheckResponse)
async def get_compliance_check(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> ComplianceCheckResponse:
    """Get the status and results of a compliance check job.

    Args:
        job_id: The unique identifier of the job.
        db: Database session.

    Returns:
        ComplianceCheckResponse with current status and results.
    """
    result = await db.execute(select(ComplianceJob).where(ComplianceJob.id == job_id))
    job = result.scalar_one_or_none()

    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return ComplianceCheckResponse(
        job_id=job.id,
        source_url=job.source_url,
        status=job.status,
        rules_applied=job.rules_applied,
        rule_results=job.rule_results,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
