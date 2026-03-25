"""
Content deduplication: prevents storing duplicate content from different URLs.
"""

import hashlib
import logging

logger = logging.getLogger(__name__)


class ContentDeduplicator:
    """
    Hash-based deduplicator that tracks content hashes to skip duplicates.
    """

    def __init__(self):
        self._seen_hashes: set[str] = set()
        self._seen_urls: set[str] = set()
        self.duplicate_count: int = 0

    def is_duplicate(self, content: str, url: str = "") -> bool:
        """
        Check if the content has been seen before.

        Args:
            content: the text content to check
            url: optional URL for logging

        Returns:
            True if duplicate, False if new.
        """
        if not content or len(content.strip()) < 20:
            return True  # Treat empty/tiny content as duplicate

        content_hash = hashlib.md5(content.encode()).hexdigest()

        if content_hash in self._seen_hashes:
            self.duplicate_count += 1
            logger.debug("Duplicate content skipped: %s (hash: %s)", url, content_hash[:8])
            return True

        self._seen_hashes.add(content_hash)
        return False

    def mark_url_seen(self, url: str):
        """Mark a URL as visited."""
        self._seen_urls.add(self._normalize_url(url))

    def is_url_seen(self, url: str) -> bool:
        """Check if a URL has already been processed."""
        return self._normalize_url(url) in self._seen_urls

    @property
    def stats(self) -> dict:
        return {
            "unique_content": len(self._seen_hashes),
            "unique_urls": len(self._seen_urls),
            "duplicates_skipped": self.duplicate_count,
        }

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL for comparison (strip trailing slash, lowercase)."""
        url = url.rstrip("/").lower()
        # Remove common tracking params
        if "?" in url:
            base, params = url.split("?", 1)
            # Keep meaningful params, drop tracking
            clean_params = []
            for p in params.split("&"):
                key = p.split("=")[0]
                if key not in ("utm_source", "utm_medium", "utm_campaign", "fbclid"):
                    clean_params.append(p)
            if clean_params:
                url = base + "?" + "&".join(clean_params)
            else:
                url = base
        return url
