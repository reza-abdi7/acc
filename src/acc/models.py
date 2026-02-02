"""Pydantic schemas for the automated compliance check service.

This module contains all data transfer objects (DTOs) and API schemas.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Dict, List

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, computed_field, constr

# Define LocaleCode as a constrained string type
LocaleCode = Annotated[str, constr(pattern=r'^[a-z]{2,3}([-_][a-zA-Z]{2})?$')]


# class SourcePayload(BaseModel):
#     """Request payload for source ingestion.

#     Attributes:
#         source_name: The name of the source.
#         source_url: The URL of the source.
#     """

#     source_name: str
#     source_url: HttpUrl


class FetchedContent(BaseModel):
    """Represents the result of fetching content from a URL.

    Attributes:
        request_url: The URL we initially tried to fetch.
        final_url: The URL after redirects, if any.
        status_code: The HTTP status code of the response.
        content_type: The MIME type of the content (e.g., 'text/html', 'application/pdf').
        content: The content of the response (e.g. text).
        is_binary: Whether the content is binary (non-text).
        binary_content_size: Size of binary content in bytes (if binary).
        storage_uri: URI reference to stored object (e.g., S3 path) for binary content.
        url_hash: SHA-256 hash of the URL (first 8 characters) used in the filename for easy tracing.
        error: Any error message if the fetch or decoding failed.
        encoding_error: Error message if decoding the text content failed.
        fetch_timestamp: Timestamp when this content was initially fetched.
    """

    request_url: str = Field(description='The URL we initially tried to fetch.')
    final_url: str | None = Field(default=None, description='The URL after redirects, if any.')
    status_code: int | None = Field(default=None, description='The HTTP status code of the response.')
    content_type: str | None = Field(
        default=None,
        description="The MIME type of the content (e.g., 'text/html', 'application/pdf').",
    )

    content: str | None = Field(default=None, description='The content of the response (e.g. text).')
    is_binary: bool = Field(default=False, description='Whether the content is binary (non-text).')
    binary_content_size: int | None = Field(
        default=None, description='Size of binary content in bytes (if binary).'
    )
    storage_uri: str | None = Field(
        default=None, description='URI reference to stored object (e.g., S3 path) for binary content.'
    )
    etag: str | None = Field(default=None, description='HTTP ETag header value')
    last_modified: str | None = Field(default=None, description='HTTP Last-Modified header value')

    url_hash: str | None = Field(
        default=None,
        description='SHA-256 hash of the URL (first 8 characters) used in the filename for easy tracing.',
    )

    error: str | None = Field(default=None, description='Any error message if the fetch or decoding failed.')
    encoding_error: str | None = Field(
        default=None, description='Error message if decoding the text content failed.'
    )

    fetch_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description='Timestamp when this content was initially fetched.',
    )

    @computed_field
    def is_success(self) -> bool:
        """Return True when fetch produced usable content and no error."""
        has_content = (self.content is not None) or (self.is_binary and self.storage_uri is not None)
        return has_content and self.error is None


class RelatedLegalDocument(BaseModel):
    """Represents a related legal document referenced in the main T&C."""

    link_text: str
    url: str
    content: str | None = None
    content_hash: str | None = None
    is_binary: bool = False
    content_type: str | None = None
    binary_size: int | None = None
    storage_uri: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    fetch_error: str | None = None

    model_config = ConfigDict(extra='forbid', strict=True)


class TermsVariant(BaseModel):
    """Represents a single locale-specific variant of the Terms & Conditions document."""

    fetched: FetchedContent
    locale: LocaleCode | None = None
    extracted_text: str | None = Field(
        default=None,
        description="Cleaned and extracted primary textual content from the T&C page's HTML.",
    )
    extracted_text_hash: str | None = Field(
        default=None, description='SHA256 hash of the extracted and processed text content.'
    )
    related_documents: List[RelatedLegalDocument] = Field(default_factory=list)
    creation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description='Timestamp when this variant was created.',
    )

    model_config = ConfigDict(extra='forbid', strict=True)

    @computed_field
    def is_binary(self) -> bool:
        """Return True when the fetched content is binary."""
        return self.fetched.is_binary

    @computed_field
    def content_type(self) -> str | None:
        """MIME type as reported by fetch."""
        return self.fetched.content_type

    @computed_field
    def storage_uri(self) -> str | None:
        """URI reference to stored binary object (if any)."""
        return self.fetched.storage_uri


class DiscoveryMethod(str, Enum):
    """How a T&C url was discovered."""

    SITEMAP = 'SITEMAP'
    ROBOTS = 'ROBOTS'
    GOOGLE_CSE = 'GOOGLE_CSE'
    HTML_EXTRACTION = 'HTML_EXTRACTION'


class DiscoveredArtifacts(BaseModel):
    """Contains the discovered T&C page URLs and content."""

    base_url: HttpUrl
    robots_txt: str | None = None
    primary_html: FetchedContent | None = None
    sitemap_urls: List[str] = Field(default_factory=list)
    terms_urls: List[str] = Field(default_factory=list)
    term_variants: List[TermsVariant] = Field(default_factory=list)
    discovery_method: DiscoveryMethod | None = Field(
        default=None, description='Method by which the T&C URL was discovered.'
    )
    error: str | None = None
    discovery_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description='Timestamp when this discovery was created.',
    )

    model_config = ConfigDict(extra='forbid', strict=True)

    @computed_field
    def terms_found(self) -> bool:
        """Return True when at least one T&C variant was found."""
        return any(v.fetched.is_success for v in self.term_variants)


class ComplianceStatus(str, Enum):
    """Enum for compliance check results."""

    APPROVED = 'APPROVED'
    BLOCKED = 'BLOCKED'
    PENDING = 'PENDING'
    ERROR = 'ERROR'


class ComplianceResult(BaseModel):
    """Response model for the compliance check result."""

    source_name: str
    source_url: HttpUrl
    status: ComplianceStatus
    rules_applied: List[str] = Field(default_factory=list)
    rule_results: Dict[str, bool] = Field(default_factory=dict)
    artifacts: DiscoveredArtifacts | None = None
    error: str | None = None

    model_config = ConfigDict(extra='forbid', strict=True)
