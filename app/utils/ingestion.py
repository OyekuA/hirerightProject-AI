import io
import ipaddress
import socket
import structlog
import threading
import docx
import httpx
import fitz
from typing import Optional
from urllib.parse import urljoin

from app.config import get_settings


logger = structlog.get_logger()

_thread_host_to_ip: threading.local = threading.local()
_original_getaddrinfo: object = socket.getaddrinfo


def _ssrf_safe_getaddrinfo(
    host: str,
    port: int,
    family: int = 0,
    type: int = 0,
    proto: int = 0,
    flags: int = 0,
) -> list:
    mapping = getattr(_thread_host_to_ip, "mapping", None)
    if mapping and host in mapping:
        host = mapping[host]
    return _original_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _ssrf_safe_getaddrinfo


def _resolve_and_validate_host(hostname: str) -> str:
    try:
        addrinfo = socket.getaddrinfo(hostname, None, family=socket.AF_UNSPEC)
    except socket.gaierror as e:
        raise ValueError(f"Cannot resolve hostname {hostname}: {e}")

    resolved_ip: Optional[str] = None
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

        if resolved_ip is None:
            resolved_ip = ip

    if resolved_ip is None:
        raise ValueError(f"Cannot resolve hostname {hostname}: no suitable address found")

    return resolved_ip


def validate_ingest_url(url: str) -> str:
    if not url.lower().startswith("https://"):
        raise ValueError("Only HTTPS URLs are allowed")

    parsed = httpx.URL(url)
    hostname = parsed.host
    if hostname is None:
        raise ValueError("Invalid URL: missing hostname")

    resolved_ip = _resolve_and_validate_host(hostname)

    logger.debug("SSRF validation passed", url=url, resolved_ip=resolved_ip)
    return resolved_ip


def validate_callback_url(url: str) -> None:
    parsed = httpx.URL(url)
    scheme = parsed.scheme
    if scheme not in ("http", "https"):
        raise ValueError("Callback URL scheme must be http or https")

    hostname = parsed.host
    if hostname is None:
        raise ValueError("Invalid URL: missing hostname")

    _resolve_and_validate_host(hostname)

    logger.debug("Callback URL SSRF validation passed", url=url)


def fetch_and_parse_cv(cv_url: str) -> str:
    resolved_ip = validate_ingest_url(cv_url)
    settings = get_settings()

    allowed_content_types = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "application/octet-stream",
    }
    allowed_extensions = {".pdf", ".docx", ".txt"}

    max_bytes = settings.MAX_INGEST_FILE_MB * 1024 * 1024
    timeout = httpx.Timeout(settings.INGEST_FETCH_TIMEOUT_SECONDS, connect=10.0)
    MAX_REDIRECTS = 5

    host_to_ip: dict[str, str] = {}
    _thread_host_to_ip.mapping = host_to_ip

    try:
        with httpx.Client(follow_redirects=False) as client:
            current_url = cv_url
            redirect_count = 0
            response = None
            final_path = ""
            content = b""

            while True:
                resolved_ip = validate_ingest_url(current_url)
                host_to_ip[httpx.URL(current_url).host] = resolved_ip

                try:
                    with client.stream("GET", current_url, timeout=timeout) as response:
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
                            resolved_ip = validate_ingest_url(current_url)
                            host_to_ip[httpx.URL(current_url).host] = resolved_ip
                            continue

                        response.raise_for_status()

                        final_path = response.url.path
                        if not any(final_path.lower().endswith(ext) for ext in allowed_extensions):
                            raise ValueError(
                                f"CV URL does not have a supported extension (final URL: {response.url}); "
                                f"supported: {allowed_extensions}"
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
    finally:
        _thread_host_to_ip.mapping = None

    if final_path:
        suffix = final_path.lower().rsplit('.', 1)[-1] if '.' in final_path else ''
        text = ""

        if suffix == 'pdf':
            try:
                doc = fitz.open(stream=content, filetype="pdf")
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                if not text.strip():
                    logger.warning("CV appears to be image-based or scanned with no extractable text layer", url="[REDACTED]")
                    raise RuntimeError("CV appears to be image-based or scanned with no extractable text layer. Please upload a text-based PDF.")
            except Exception as e:
                logger.error("PDF parsing failed", url="[REDACTED]", error=str(e))
                raise RuntimeError(f"PDF parsing failed: {e}") from e

        elif suffix == 'docx':
            try:
                document = docx.Document(io.BytesIO(content))
                text = "\n".join(paragraph.text for paragraph in document.paragraphs)
                if not text.strip():
                    logger.warning("CV file appears to have no extractable text", url="[REDACTED]", path=final_path)
                    raise RuntimeError("CV file appears to have no extractable text. Please upload a text-based docx file.")
            except Exception as e:
                logger.error("DOCX parsing failed", url="[REDACTED]", error=str(e))
                raise RuntimeError(f"DOCX parsing failed: {e}") from e

        elif suffix == 'txt':
            text = content.decode("utf-8", errors="replace")
            if not text.strip():
                logger.warning("CV file appears to have no extractable text", url="[REDACTED]", path=final_path)
                raise RuntimeError("CV file appears to have no extractable text. Please upload a text-based txt file.")

        else:
            logger.error("Unsupported file type for CV", url="[REDACTED]", path=final_path)
            raise ValueError(f"Unsupported file type: only PDF, DOCX, and TXT are supported (file: {final_path})")

        logger.debug("CV parsed successfully", url="[REDACTED]", text_length=len(text))
        return text

    raise RuntimeError("Failed to download CV: no successful response received")


def truncate_to_prompt_cap(text: str) -> str:
    settings = get_settings()
    max_chars = settings.MAX_PROMPT_CHARS
    if len(text) > max_chars:
        logger.warning("Prompt text truncated", original_length=len(text), cap=max_chars)
        return text[:max_chars]
    return text
