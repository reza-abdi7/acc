"""Application configuration loaded from environment variables."""

import logging
from typing import Any, Dict

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    # HTTP client settings
    fetch_timeout: float = Field(default=10.0, description='Timeout in seconds for HTTP requests.')
    max_retries: int = Field(default=3, description='Maximum number of retries for failed HTTP requests.')
    default_headers: Dict[str, Any] = Field(
        default={}, description='Default headers for HTTP requests (as JSON string in .env).'
    )
    requests_per_second: float = Field(
        default=10.0, description='Maximum number of requests per second to the same domain.'
    )
    max_connections: int = Field(default=2000, description='Maximum number of connections to keep open.')
    max_keepalive_pool: int = Field(
        default=2000, description='Maximum number of keep-alive connections to keep open.'
    )

    # Google Custom Search API
    google_cse_api_key: str | None = Field(default=None, description='Google Custom Search API key.')
    google_cse_id: str | None = Field(default=None, description='Google Custom Search Engine ID.')

    # Storage settings
    storage_mode: str = Field(default='local', description='Storage mode: "local" or "s3".')
    local_storage_path: str = Field(default='./data/storage', description='Base directory for local storage.')

    # S3 settings
    s3_bucket_name: str = Field(default='compliance-artifacts', description='S3 bucket name.')
    aws_region: str = Field(default='eu-north-1', description='AWS region for S3.')
    s3_endpoint_url: str | None = Field(default=None, description='Optional S3 endpoint URL.')
    aws_access_key_id: str | None = Field(default=None, description='AWS access key ID.')
    aws_secret_access_key: str | None = Field(default=None, description='AWS secret access key.')

    # Selenium settings
    selenium_remote_url: str = Field(
        default='http://stat-hh-smiworker:4441', description='Remote Selenium Grid URL.'
    )

    # Database settings
    postgres_user: str = Field(default='postgres', description='Postgres username.')
    postgres_password: str = Field(default='postgres', description='Postgres password.')
    postgres_server: str = Field(default='localhost', description='Postgres server host.')
    postgres_port: int = Field(default=5432, description='Postgres port.')
    postgres_db: str = Field(default='acc_db', description='Postgres database name.')

    @property
    def postgres_dsn(self) -> str:
        """Construct the Postgres DSN from individual settings."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_server}:{self.postgres_port}/{self.postgres_db}"

    # Redis settings
    redis_host: str = Field(default='localhost', description='Redis host.')
    redis_port: int = Field(default=6379, description='Redis port.')

    @property
    def redis_url(self) -> str:
        """Construct the Redis URL."""
        return f"redis://{self.redis_host}:{self.redis_port}"


try:
    settings = Settings()
    logger.info('Application settings loaded successfully.')
except ValidationError as e:
    logger.critical(f'Failed to load application settings: {e}', exc_info=True)
    raise SystemExit(f'Configuration Error: {e}') from e
