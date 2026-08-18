"""
Mwalimu AI - Secure File and Configuration Handling

Security responsibilities:
- API key validation without exposing the key
- Upload validation
- Safe temporary-file handling
- Filename sanitisation
- Sensitive-data cleanup helpers

IMPORTANT:
API keys must be supplied through environment variables or deployment
secrets. Never place an API key directly in this file or app.py.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Optional


ALLOWED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

DEFAULT_MAX_UPLOAD_MB = 25


def get_openai_api_key() -> str:
    """
    Retrieve the OpenAI API key from the deployment environment.

    The actual key is never printed or returned to the UI.
    """

    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. "
            "Add it to the application's secure secrets/environment "
            "variables rather than placing it in source code."
        )

    return api_key


def validate_file(
    filename: str,
    file_bytes: bytes,
    max_size_mb: int = DEFAULT_MAX_UPLOAD_MB,
) -> None:
    """
    Validate an uploaded examination document before processing.
    """

    if not filename:
        raise ValueError("A filename is required.")

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if len(file_bytes) > max_size_mb * 1024 * 1024:
        raise ValueError(
            f"The uploaded file exceeds the "
            f"{max_size_mb} MB limit."
        )

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ValueError(
            f"Unsupported file type '{extension}'. "
            f"Allowed types: {allowed}"
        )


def safe_filename(filename: str) -> str:
    """
    Remove unsafe path characters from an uploaded filename.

    This prevents an uploaded filename from being interpreted as a
    filesystem path.
    """

    name = Path(filename).name

    name = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        name,
    )

    name = name.strip(".")

    return name or "uploaded_document"


def create_secure_temp_file(
    data: bytes,
    suffix: str = ".bin",
) -> str:
    """
    Create a temporary file containing supplied data.

    The caller is responsible for deleting the returned file as soon
    as processing is complete.
    """

    fd, path = tempfile.mkstemp(
        suffix=suffix,
    )

    try:
        with os.fdopen(fd, "wb") as temp_file:
            temp_file.write(data)

        return path

    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass

        raise


def secure_delete(path: Optional[str]) -> None:
    """
    Delete a temporary file if it exists.

    Failure to delete a non-existent file is ignored.
    """

    if not path:
        return

    try:
        file_path = Path(path)

        if file_path.exists() and file_path.is_file():
            file_path.unlink()

    except OSError:
        # Do not expose filesystem details to the user.
        pass


def redact_error_message(error: Exception) -> str:
    """
    Return a safer error message for UI logging.

    API keys, tokens and authorization headers should never be exposed
    in user-facing error messages.
    """

    message = str(error)

    sensitive_patterns = [
        r"sk-[A-Za-z0-9_-]+",
        r"Bearer\s+[A-Za-z0-9._-]+",
        r"api[_-]?key[=:]\s*[A-Za-z0-9._-]+",
    ]

    for pattern in sensitive_patterns:
        message = re.sub(
            pattern,
            "[REDACTED]",
            message,
            flags=re.IGNORECASE,
        )

    return message


def is_safe_filename(filename: str) -> bool:
    """
    Basic path-traversal protection for uploaded filenames.
    """

    if not filename:
        return False

    path = Path(filename)

    if path.name != filename:
        return False

    dangerous_parts = {
        "..",
        "/",
        "\\",
    }

    return not any(
        part in filename
        for part in dangerous_parts
    )
