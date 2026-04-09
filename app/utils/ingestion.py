"""Ingestion‑specific utility functions (SSRF validation, CV parsing, prompt truncation)."""

import ipaddress
import socket
import structlog
import httpx
import fitz
from urllib.parse import urljoin

from app.config import get_settings


logger = structlog.get_logger()


def validate_ingest_url(url: str) -> None:
    """Raise ValueError if the URL is not allowed (SSRF / scheme / private network).

    Rules:
      - Only https:// scheme permitted.
      - Resolved IP must not be in private, loopback, link‑local, reserved,
        multicast, or unspecified ranges.
      - DNS is re‑resolved each call.
    """
    if not url.lower().startswith("https://"):
        raise ValueError("Only HTTPS URLs are allowed")

    parsed = httpx.URL(url)
    hostname = parsed.host
    if hostname is None:
        raise ValueError("Invalid URL: missing hostname")

    try:
        addrinfo = socket.getaddrinfo(hostname, None, family=socket.AF_UNSPEC)
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve hostname {hostname}: {e}")

    for family, _, _, _, sockaddr in addrinfo:
        if family == socket.AF_INET:
            ip = sockaddr[0]
        elif family == socket.AF_INET6:
            ip = sockaddr[0].split("%")[0]
        else:
            continue

        ip_obj = ipaddress.ip_address(ip)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        ):
            raise ValueError(f"Resolved IP {ip} is in a prohibited network range")

    logger.debug("SSRF validation passed", url=url)


def fetch_and_parse_cv(cv_url: str) -> str:
    """Download a CV (PDF) and extract plain text.

    Args:
        cv_url: HTTPS URL to the CV document.

    Returns:
        Extracted plain text.

    Raises:
        ValueError: SSRF validation fails, unsupported Content‑Type / extension,
                    or file size exceeds MAX_INGEST_FILE_MB.
        httpx.HTTPError: Download failed.
        RuntimeError: Parsing failed.
    """
    validate_ingest_url(cv_url)
    settings = get_settings()

    allowed_content_types = {
        "application/pdf",
        "application/octet-stream",
    }
    allowed_extensions = {".pdf"}


    max_bytes = settings.MAX_INGEST_FILE_MB * 1024 * 1024
    timeout = httpx.Timeout(settings.INGEST_FETCH_TIMEOUT_SECONDS, connect=10.0)
    MAX_REDIRECTS = 5

    with httpx.Client(follow_redirects=False) as client:
        current_url = cv_url
        redirect_count = 0
        response = None

        while True:
            validate_ingest_url(current_url)

            try:
                with client.stream("GET", current_url, timeout=timeout) as response:
                    response.raise_for_status()

                    if response.status_code in (301, 302, 303, 307, 308):
                        if redirect_count >= MAX_REDIRECTS:
                            raise ValueError(
                                f"Too many redirects (max {MAX_REDIRECTS})"
                            )
                        location = response.headers.get("Location")
                        if not location:
                            raise ValueError("Redirect response missing Location header")
                        resolved = urljoin(str(response.url), location)
                        current_url = str(resolved)
                        redirect_count += 1
                        continue

                    validate_ingest_url(str(response.url))

                    final_path = response.url.path
                    if not any(final_path.lower().endswith(ext) for ext in allowed_extensions):
                        raise ValueError(
                            f"CV URL does not have a .pdf extension (final URL: {response.url})"
                        )

                    content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
                    if content_type not in allowed_content_types:
                        raise ValueError(
                            f"Unsupported Content‑Type {content_type}; "
                            f"must be one of {allowed_content_types}"
                        )

                    bytes_read = 0
                    chunks = []
                    for chunk in response.iter_bytes():
                        bytes_read += len(chunk)
                        if bytes_read > max_bytes:
                            raise ValueError(
                                f"CV exceeds size limit ({settings.MAX_INGEST_FILE_MB} MiB)"
                            )
                        chunks.append(chunk)

                    content = b"".join(chunks)
                    break

            except httpx.HTTPError as e:
                logger.error("HTTP error downloading CV", url="[REDACTED]", error=str(e))
                raise RuntimeError(f"Failed to download CV: {e}") from e

    is_pdf = final_path.lower().endswith('.pdf')
    text = ""
    if is_pdf:
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
        except Exception as e:
            logger.error("PDF parsing failed", url="[REDACTED]", error=str(e))
            raise RuntimeError(f"PDF parsing failed: {e}") from e
    else:
        logger.error("Unsupported file type for CV", url="[REDACTED]", path=final_path)
        raise ValueError(f"Unsupported file type: only PDF is supported (file: {final_path})")

    logger.debug("CV parsed successfully", url="[REDACTED]", text_length=len(text))
    return text


def truncate_to_prompt_cap(text: str) -> str:
    """Silently truncate a text to the maximum allowed prompt length.

    The maximum length is read from the application settings (MAX_PROMPT_CHARS).
    If the text is longer than the limit, it is truncated without raising an
    exception and without logging. Shorter texts are returned unchanged.

    Args:
        text: The input text that may be passed to Gemini.

    Returns:
        The same text if within the limit, otherwise a truncated version
        containing the first MAX_PROMPT_CHARS characters.
    """
    settings = get_settings()
    max_chars = settings.MAX_PROMPT_CHARS
    return text[:max_chars]