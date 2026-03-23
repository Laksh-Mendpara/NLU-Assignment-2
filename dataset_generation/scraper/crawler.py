"""
Async BFS web crawler for the iitj.ac.in domain ecosystem.

Manages the crawl frontier, visited set, link discovery, and
dispatches to html_extractor / pdf_extractor as appropriate.
"""

import asyncio
import json
import logging
import os
import re
import time
from urllib.parse import urljoin, urlparse, urldefrag

import aiohttp
import aiofiles

from config.settings import (
    ALLOWED_DOMAIN,
    DEFAULT_DELAY,
    EXCLUDED_PATTERNS,
    MAX_CONCURRENCY,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    RETRY_BACKOFF_BASE,
    STATE_FILE,
    USER_AGENT,
)
from scraper.html_extractor import extract_content
from scraper.pdf_extractor import download_and_extract
from scraper.metadata import build_metadata
from scraper.dedup import ContentDeduplicator

logger = logging.getLogger(__name__)

# Compiled excluded patterns
_EXCLUDED_RE = [re.compile(p, re.IGNORECASE) for p in EXCLUDED_PATTERNS]


class IITJCrawler:
    """
    Async BFS crawler for iitj.ac.in.
    """

    def __init__(
        self,
        output_dir: str,
        max_pages: int = 0,
        delay: float = DEFAULT_DELAY,
        concurrency: int = MAX_CONCURRENCY,
        resume: bool = False,
    ):
        self.output_dir = output_dir
        self.max_pages = max_pages  # 0 = unlimited
        self.delay = delay
        self.concurrency = concurrency
        self.resume = resume

        self.frontier: asyncio.Queue = asyncio.Queue()
        self.visited: set[str] = set()
        self.dedup = ContentDeduplicator()

        self.pages_crawled = 0
        self.pages_failed = 0
        self.pdfs_extracted = 0
        self.start_time = 0.0

        self._data_dir = os.path.join(output_dir, "data")
        self._pdf_dir = os.path.join(output_dir, "pdfs")
        self._state_path = os.path.join(output_dir, STATE_FILE)

    async def initialize(self, seed_urls: list[str]):
        """Set up output dirs and seed the frontier."""
        os.makedirs(self._data_dir, exist_ok=True)
        os.makedirs(self._pdf_dir, exist_ok=True)

        # Resume from saved state
        if self.resume and os.path.exists(self._state_path):
            await self._load_state()
            logger.info("Resumed: %d visited, %d in frontier", len(self.visited), self.frontier.qsize())
        else:
            for url in seed_urls:
                normalized = self._normalize_url(url)
                if normalized not in self.visited:
                    await self.frontier.put(normalized)

        logger.info("Crawler initialized. Frontier size: %d", self.frontier.qsize())

    async def crawl(self):
        """Run the BFS crawl with a pool of async workers."""
        self.start_time = time.time()

        connector = aiohttp.TCPConnector(limit=self.concurrency, ssl=False)
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        ) as session:
            # Create worker tasks
            workers = [
                asyncio.create_task(self._worker(session, i))
                for i in range(self.concurrency)
            ]

            # Wait until frontier is empty and all workers are done
            await self.frontier.join()

            # Cancel workers
            for w in workers:
                w.cancel()

            await asyncio.gather(*workers, return_exceptions=True)

        elapsed = time.time() - self.start_time
        logger.info(
            "Crawl complete. Pages: %d, PDFs: %d, Failed: %d, Time: %.1fs",
            self.pages_crawled, self.pdfs_extracted, self.pages_failed, elapsed,
        )
        logger.info("Dedup stats: %s", self.dedup.stats)

    async def _worker(self, session: aiohttp.ClientSession, worker_id: int):
        """Worker that processes URLs from the frontier."""
        while True:
            try:
                url = await self.frontier.get()
            except asyncio.CancelledError:
                return

            try:
                # Check max pages limit
                if self.max_pages > 0 and self.pages_crawled >= self.max_pages:
                    self.frontier.task_done()
                    continue

                if url in self.visited:
                    self.frontier.task_done()
                    continue

                self.visited.add(url)

                # Process the URL
                await self._process_url(url, session)

                # Polite delay
                await asyncio.sleep(self.delay)

                # Periodic state save
                if self.pages_crawled % 50 == 0 and self.pages_crawled > 0:
                    await self._save_state()
                    self._log_progress()

            except asyncio.CancelledError:
                self.frontier.task_done()
                return
            except Exception as e:
                logger.error("[Worker %d] Error processing %s: %s", worker_id, url, e)
                self.pages_failed += 1
            finally:
                self.frontier.task_done()

    async def _process_url(self, url: str, session: aiohttp.ClientSession):
        """Process a single URL: fetch, extract, save."""

        # Skip excluded patterns
        if self._is_excluded(url):
            logger.debug("Excluded: %s", url)
            return

        is_pdf = url.lower().endswith(".pdf")

        if is_pdf:
            await self._process_pdf(url, session)
        else:
            await self._process_html(url, session)

    async def _process_html(self, url: str, session: aiohttp.ClientSession):
        """Fetch HTML page, extract content, discover links."""
        html = await self._fetch_with_retry(url, session)
        if html is None:
            return

        # Extract content
        extracted = extract_content(html, url)
        content = extracted["content"]

        # Discover and enqueue new links
        await self._discover_links(html, url)

        # Skip if content is empty or duplicate
        if not content or self.dedup.is_duplicate(content, url):
            return

        # Build metadata and save
        meta = build_metadata(
            url=url,
            title=extracted["title"],
            description=extracted["description"],
            content=content,
        )

        from config.settings import DEFAULT_DOC_TYPE
        if meta.get("doc_type") == DEFAULT_DOC_TYPE:
            logger.debug("Skipped saving %s (doc_type: %s)", url, DEFAULT_DOC_TYPE)
            return

        await self._save_document(meta, content)
        self.pages_crawled += 1

        logger.info(
            "[%d] Scraped: %s (%d chars)",
            self.pages_crawled, url, len(content),
        )

    async def _process_pdf(self, url: str, session: aiohttp.ClientSession):
        """Download and extract a PDF."""
        result = await download_and_extract(url, session)
        if result is None:
            return

        content = result["content"]
        if self.dedup.is_duplicate(content, url):
            return

        meta = build_metadata(
            url=url,
            title=result["metadata"].get("filename", ""),
            content=content,
            doc_type=None, # Auto-classify instead of hardcoding PDF Document
            extra=result["metadata"],
        )

        from config.settings import DEFAULT_DOC_TYPE
        if meta.get("doc_type") == DEFAULT_DOC_TYPE:
            logger.debug("Skipped saving PDF %s (doc_type: %s)", url, DEFAULT_DOC_TYPE)
            return

        await self._save_document(meta, content, is_pdf=True)
        self.pdfs_extracted += 1
        self.pages_crawled += 1

        logger.info(
            "[%d] PDF extracted: %s (%d pages, %d chars)",
            self.pages_crawled, url,
            result["metadata"].get("pages", 0), len(content),
        )

    async def _discover_links(self, html: str, base_url: str):
        """Extract links from HTML and add new ones to frontier."""
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href or href.startswith("#"):
                continue

            # Resolve relative URL
            full_url = urljoin(base_url, href)
            # Remove fragment
            full_url, _ = urldefrag(full_url)

            # Some pdf links have URL encodings (e.g. %20 for spaces)
            # We normalize them but keep them properly parseable
            full_url = full_url.replace(" ", "%20")
            from urllib.parse import unquote
            # Unquote it to check if it's a PDF natively, then requote properly if needed,
            # but aiohttp typically handles raw percent-encodings fine if preserved.
            normalized = self._normalize_url(full_url)

            # Check domain
            if not self._is_allowed_domain(normalized):
                continue

            # Check excluded
            if self._is_excluded(normalized):
                continue

            # Add to frontier if new
            if normalized not in self.visited:
                await self.frontier.put(normalized)

    async def _fetch_with_retry(self, url: str, session: aiohttp.ClientSession) -> str | None:
        """Fetch a URL with exponential backoff retries."""
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        content_type = resp.headers.get("Content-Type", "")
                        if "pdf" in content_type.lower():
                            # It's actually a PDF, process it as such
                            # Re-add to frontier for PDF processing
                            return None
                        return await resp.text(errors="replace")
                    elif resp.status == 404:
                        logger.debug("404 Not Found: %s", url)
                        return None
                    elif resp.status == 429:
                        # Rate limited - back off more aggressively
                        wait = RETRY_BACKOFF_BASE ** (attempt + 2)
                        logger.warning("Rate limited on %s, waiting %.1fs", url, wait)
                        await asyncio.sleep(wait)
                    else:
                        logger.debug("HTTP %d: %s", resp.status, url)
                        if resp.status >= 500:
                            await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)
                            continue
                        return None
            except asyncio.TimeoutError:
                logger.debug("Timeout (attempt %d): %s", attempt + 1, url)
                await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)
            except aiohttp.ClientError as e:
                logger.debug("Client error (attempt %d) for %s: %s", attempt + 1, url, e)
                await asyncio.sleep(RETRY_BACKOFF_BASE ** attempt)
            except Exception as e:
                logger.error("Unexpected error fetching %s: %s", url, e)
                return None

        self.pages_failed += 1
        logger.warning("Failed after %d retries: %s", MAX_RETRIES, url)
        return None

    async def _save_document(self, metadata: dict, content: str, is_pdf: bool = False):
        """Save a scraped document as a JSON file."""
        doc_id = metadata["id"]
        target_dir = self._pdf_dir if is_pdf else self._data_dir
        filepath = os.path.join(target_dir, f"{doc_id}.json")

        doc = {
            "metadata": metadata,
            "content": content,
        }

        async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
            await f.write(json.dumps(doc, ensure_ascii=False, indent=2))

    async def _save_state(self):
        """Save crawl state for resume capability."""
        state = {
            "visited": list(self.visited),
            "pages_crawled": self.pages_crawled,
            "pdfs_extracted": self.pdfs_extracted,
            "pages_failed": self.pages_failed,
        }
        async with aiofiles.open(self._state_path, "w") as f:
            await f.write(json.dumps(state))
        logger.debug("State saved: %d visited URLs", len(self.visited))

    async def _load_state(self):
        """Load crawl state from file."""
        try:
            async with aiofiles.open(self._state_path, "r") as f:
                state = json.loads(await f.read())
            self.visited = set(state.get("visited", []))
            self.pages_crawled = state.get("pages_crawled", 0)
            self.pdfs_extracted = state.get("pdfs_extracted", 0)
            self.pages_failed = state.get("pages_failed", 0)
        except Exception as e:
            logger.error("Failed to load state: %s", e)

    def _log_progress(self):
        """Log crawl progress."""
        elapsed = time.time() - self.start_time
        rate = self.pages_crawled / elapsed if elapsed > 0 else 0
        logger.info(
            "Progress: %d pages, %d PDFs, %d failed, %d queued, %.1f pages/min",
            self.pages_crawled, self.pdfs_extracted, self.pages_failed,
            self.frontier.qsize(), rate * 60,
        )

    async def get_manifest(self) -> list[dict]:
        """Generate a manifest of all scraped documents."""
        manifest = []
        for filename in sorted(os.listdir(self._data_dir)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self._data_dir, filename)
            try:
                async with aiofiles.open(filepath, "r") as f:
                    doc = json.loads(await f.read())
                manifest.append(doc["metadata"])
            except Exception:
                continue
        return manifest

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL for dedup (strip trailing slash, fragment)."""
        url, _ = urldefrag(url)
        url = url.rstrip("/")
        return url

    @staticmethod
    def _is_allowed_domain(url: str) -> bool:
        """Check if URL belongs to iitj.ac.in domain."""
        try:
            host = urlparse(url).hostname
            return host is not None and host.endswith(ALLOWED_DOMAIN)
        except Exception:
            return False

    @staticmethod
    def _is_excluded(url: str) -> bool:
        """Check if URL matches any exclusion pattern."""
        for pattern in _EXCLUDED_RE:
            if pattern.search(url):
                return True
        return False
