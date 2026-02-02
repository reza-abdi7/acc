"""Storage service for the automated compliance check system.

This module provides functionality to store binary content (like documents, PDFs)
either in cloud storage (S3) or locally on disk, depending on configuration.
"""

import hashlib
import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import boto3 conditionally to avoid errors if AWS SDK is not installed
# try:
#     import boto3
#     from botocore.exceptions import ClientError
#     BOTO3_AVAILABLE = True
# except ImportError:
#     BOTO3_AVAILABLE = False
from pydantic import validate_call

from src.acc.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Service to handle storing and retrieving binary content in storage.

    This service provides a consistent interface for storing and retrieving content,
    whether using local file storage or cloud storage (S3).

    Attributes:
        storage_mode: Whether to use 'local' or 's3' storage.
        local_storage_path: Base directory for local file storage.
        bucket_name: The name of the S3 bucket (when using S3).
        s3_client: The boto3 S3 client (when using S3).
    """

    def __init__(
        self,
        storage_mode: Optional[str] = None,
        local_storage_path: Optional[str] = None,
        bucket_name: Optional[str] = None,
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ):
        """Initialize the storage service.

        Args:
            storage_mode: Storage mode to use ('local' or 's3'). Defaults to settings.storage_mode.
            local_storage_path: Path for local file storage. Defaults to settings.local_storage_path.
            bucket_name: S3 bucket name (for S3 mode). Defaults to settings.s3_bucket_name.
            region_name: AWS region (for S3 mode). Defaults to settings.aws_region.
            endpoint_url: S3 endpoint URL (for S3 mode). Defaults to settings.s3_endpoint_url.
        """
        # Get storage mode - default to 'local' if not specified
        self.storage_mode = storage_mode or getattr(settings, 'storage_mode', 'local')

        if self.storage_mode == 's3':
            # TODO: S3 implementation is currently on hold. The following S3-specific initialization is bypassed.
            # Future implementation should restore this block and remove the forced fallback to 'local'.
            logger.warning(
                "S3 storage mode is configured, but S3 implementation is currently on hold. Falling back to 'local' storage."
            )
            self.storage_mode = 'local'
            self.bucket_name = (
                None  # Ensure S3 specific attributes are not set if falling back
            )
            self.region_name = None
            self.endpoint_url = None
            self.s3_client = None

            # Original S3 initialization logic (commented out):
            # if not BOTO3_AVAILABLE:
            #     logger.warning("boto3 not available, falling back to local storage")
            #     self.storage_mode = 'local'
            # else:
            #     # Use settings or override with provided parameters
            #     self.bucket_name = bucket_name or settings.s3_bucket_name
            #     self.region_name = region_name or settings.aws_region
            #     self.endpoint_url = endpoint_url or settings.s3_endpoint_url
            #
            #     # Initialize S3 client
            #     self.s3_client = boto3.client(
            #         's3',
            #         region_name=self.region_name,
            #         endpoint_url=self.endpoint_url,
            #         aws_access_key_id=settings.aws_access_key_id,
            #         aws_secret_access_key=settings.aws_secret_access_key
            #     )
            #     logger.info(f"Initialized storage service in S3 mode with bucket: {self.bucket_name}")

        if self.storage_mode == 'local':
            # Set up local storage path - don't create directory as it's a shared drive
            # Determine the path string explicitly to satisfy mypy
            path_val: Optional[str] = local_storage_path

            if path_val is None:
                # Try to get from settings; getattr's default is if attr itself is missing.
                # If settings.local_storage_path can be None, that's handled next.
                path_from_settings = getattr(settings, 'local_storage_path', None)
                if path_from_settings is not None:
                    path_val = path_from_settings

            if path_val is None:  # If still None, use the final default
                path_val = 'data/storage'

            self.local_storage_path = Path(path_val)
            logger.info(
                f'Initialized storage service in local mode with path: {self.local_storage_path}'
            )

    @validate_call
    async def store_binary_content(
        self, content: bytes, source_url: str, content_type: str, purpose: str = 'legal'
    ) -> tuple[str, str]:
        """Store binary content and return a URI reference and URL hash.

        Stores content either locally or in S3 based on the storage_mode.

        Args:
            content: The binary content to store.
            source_url: The source URL the content was fetched from.
            content_type: The MIME type of the content.
            purpose: The purpose of the content, used to determine storage location.
                     Options: 'legal' (default) for T&C docs, 'source' for original source files,
                     or custom value for other types.

        Returns:
            A tuple containing:
                - URI string pointing to the stored object (file:// or s3:// URI)
                - URL hash string (first 8 characters of SHA256 hash of source_url)

        Raises:
            Exception: If storing the content fails.
        """
        try:
            # Generate a deterministic but unique object key
            # content hash makes sure we don't store duplicate files with identical content even if they
            # come from different url
            content_hash = hashlib.sha256(content).hexdigest()
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            # url hash will help to trace the file back to its source without storing the entire url which is
            # useful for debugging, grouping related docs from the same source and track when the same url
            # provides different contents over time
            url_hash = hashlib.sha256(source_url.encode()).hexdigest()[:8]

            # Extract file extension from content type
            extension = self._get_extension_from_content_type(content_type)

            # Create filename in the format: YYYYMMDDHHMMSS_URL-HASH_CONTENT-HASH.ext
            filename = f'{timestamp}_{url_hash}_{content_hash[:8]}{extension}'

            if self.storage_mode == 's3':
                # TODO: Implement S3 storage
                # # Create S3 object key with structured path by year/month/day/hour
                # folder_path = datetime.now().strftime("%Y/%m/%d/%H/")
                # full_object_key = folder_path + filename
                #
                # # Upload to S3
                # self.s3_client.upload_fileobj(
                #     io.BytesIO(content),
                #     self.bucket_name,
                #     full_object_key,
                #     ExtraArgs={
                #         'ContentType': content_type
                #     }
                # )
                #
                # # Return an S3 URI and the URL hash
                # uri = f"s3://{self.bucket_name}/{full_object_key}"
                # logger.info(f"Stored content at: {uri} (S3)")
                # # Return both the URI and the URL hash
                # return uri, url_hash

                # For now, return a placeholder URI and the URL hash
                uri = f'file:///{filename}'  # Placeholder for local storage or future S3
                logger.info(
                    f'Skipping S3 storage. Content hash: {url_hash}. Placeholder URI: {uri}'
                )
                return uri, url_hash
            else:  # local storage
                # For local storage, determine the appropriate base folder based on purpose
                if purpose == 'legal':
                    # Use the path directly as configured for legal documents
                    folder = self.local_storage_path
                else:
                    # For other purposes (like 'source'), create a subfolder under the base path
                    # Use the parent directory of the legal folder
                    base_folder = (
                        self.local_storage_path.parent
                        if hasattr(self.local_storage_path, 'parent')
                        else Path(str(self.local_storage_path)).parent
                    )
                    folder = base_folder / purpose

                    # Ensure the folder exists, but only log if it doesn't (no creation)
                    if not folder.exists():
                        logger.warning(
                            f"Purpose folder '{purpose}' does not exist at {folder}. Using base path instead."
                        )
                        folder = base_folder

                # Create the full path for the file
                full_path = folder / filename

                # Write content to file
                with open(full_path, 'wb') as f:
                    f.write(content)

                # Return a file URI
                uri = f'file://{full_path.absolute()}'
                logger.info(f'Stored content at: {uri} (local, purpose: {purpose})')
                # Return both the URI and the URL hash
                return uri, url_hash

        except Exception as e:
            error_msg = f'Failed to store binary content from {source_url}'
            logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e

    def _get_extension_from_content_type(self, content_type: str) -> str:
        """Get the appropriate file extension based on MIME type.

        Uses Python's built-in mimetypes module to determine the appropriate
        file extension for a given MIME type.

        Args:
            content_type: The MIME type string.

        Returns:
            A string containing the file extension including the dot (e.g., '.pdf').
            Returns '.bin' if no appropriate extension is found.
        """
        # Initialize mimetypes if needed
        if not mimetypes.inited:
            mimetypes.init()

        # Clean the content type by removing parameters (like charset)
        # E.g., 'text/html; charset=utf-8' -> 'text/html'
        if ';' in content_type:
            content_type = content_type.split(';', 1)[0].strip()

        # Handle common case differences and content-type variations
        content_type = content_type.lower()

        # Get extension using mimetypes
        ext = mimetypes.guess_extension(content_type)

        # Return the extension if found, otherwise default to .bin
        return ext if ext else '.bin'


# Create a singleton instance
storage_service = StorageService()
