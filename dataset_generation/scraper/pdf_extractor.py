"""
PDF extractor: downloads PDFs and converts them to Markdown using pymupdf4llm.
"""

import logging
import os
import tempfile
from urllib.parse import urlparse, unquote

import aiohttp
import pymupdf4llm

from config.settings import MAX_PDF_SIZE_MB, USER_AGENT, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


async def download_and_extract(url: str, session: aiohttp.ClientSession) -> dict | None:
    """
    Download a PDF from a URL and extract its content as Markdown.

    Returns:
        dict with keys: 'content' (str), 'metadata' (dict with pages, file_size, filename)
        or None if extraction fails.
    """
    filename = _get_filename(url)

    try:
        # Stream download to check size first
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT * 3),
            headers={"User-Agent": USER_AGENT},
        ) as resp:
            if resp.status != 200:
                logger.warning("PDF download failed (status %d): %s", resp.status, url)
                return None

            # Check content length
            content_length = resp.headers.get("Content-Length")
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > MAX_PDF_SIZE_MB:
                    logger.warning(
                        "PDF too large (%.1f MB > %d MB limit): %s",
                        size_mb, MAX_PDF_SIZE_MB, url,
                    )
                    return None

            # Download to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = tmp.name
                total_bytes = 0
                async for chunk in resp.content.iter_chunked(8192):
                    total_bytes += len(chunk)
                    # Double-check size during download
                    if total_bytes > MAX_PDF_SIZE_MB * 1024 * 1024:
                        logger.warning("PDF exceeded size limit during download: %s", url)
                        os.unlink(tmp_path)
                        return None
                    tmp.write(chunk)

        # Extract with pymupdf (fitz) directly for pure text, skipping all images/OCR
        import pymupdf
        try:
            doc = pymupdf.open(tmp_path)
            md_text = ""
            for page in doc:
                md_text += page.get_text("text") + "\n\n"
            page_count = len(doc)
            doc.close()
        except Exception as e:
            logger.error("pymupdf text extraction failed for %s: %s", url, e)
            os.unlink(tmp_path)
            return None

        # Cleanup
        os.unlink(tmp_path)

        if not md_text or len(md_text.strip()) < 20:
            logger.warning("PDF extraction yielded empty/tiny content: %s", url)
            return None

        import langid
        lang, _ = langid.classify(md_text)
        if lang != 'en':
            logger.info("PDF extraction skipped non-English content (%s): %s", lang, url)
            return None

        return {
            "content": md_text.strip(),
            "metadata": {
                "pages": page_count,
                "file_size_bytes": total_bytes,
                "filename": filename,
            },
        }

    except aiohttp.ClientError as e:
        logger.error("Network error downloading PDF %s: %s", url, e)
    except Exception as e:
        logger.error("Unexpected error processing PDF %s: %s", url, e)

    # Cleanup temp file if it exists
    try:
        if "tmp_path" in locals():
            os.unlink(tmp_path)
    except OSError:
        pass

    return None


def _get_filename(url: str) -> str:
    """Extract a human-readable filename from a PDF URL."""
    path = urlparse(url).path
    name = os.path.basename(unquote(path))
    return name if name else "unknown.pdf"
