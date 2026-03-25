from __future__ import annotations

"""Simple scraper script used in Q1."""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add directory to sys path to avoid import errors when running from root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    DEFAULT_DELAY,
    DEFAULT_OUTPUT_DIR,
    FOCUSED_SEED_URLS,
    KNOWN_SUBDOMAINS,
    MANIFEST_FILE,
    MAX_CONCURRENCY,
    SEED_URLS,
    SITEMAP_URL,
)


def setup_logging(verbose: bool = False):
    """Configure logging with colored output."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s │ %(levelname)-7s │ %(message)s"
    datefmt = "%H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    # Suppress noisy loggers
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("trafilatura").setLevel(logging.WARNING)
    logging.getLogger("pymupdf").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    # I kept the settings in one place so the script is easy to read.
    return argparse.Namespace(
        output_dir=DEFAULT_OUTPUT_DIR,
        max_pages=2400,
        delay=0.4,
        concurrency=6,
        resume=False,
        verbose=False,
        skip_sitemap=False,
        skip_atoz=False,
        focused_seeds_only=True,
        seed_url=[],
        seed_file=None,
    )


def load_extra_seed_urls(args: argparse.Namespace) -> list[str]:
    extra_seed_urls = list(args.seed_url or [])
    if args.seed_file:
        seed_file = Path(args.seed_file)
        if seed_file.exists():
            # Allow adding a small custom seed list without editing the code.
            extra_seed_urls.extend(
                line.strip() for line in seed_file.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        else:
            logging.warning("Seed file not found: %s", seed_file)
    return extra_seed_urls


def rebuild_output_artifacts(output_dir: str, crawl_stats: dict | None = None) -> dict:
    """Rebuild manifest.json and data.txt from already-saved document JSON files."""
    from scraper.content_filters import sanitize_document

    output_path = Path(output_dir)
    manifest_path = output_path / MANIFEST_FILE
    documents: list[dict] = []
    total_written = 0
    skipped_documents = 0

    for directory_name, storage_kind in (("data", "html"), ("pdfs", "pdf")):
        directory = output_path / directory_name
        if not directory.is_dir():
            continue
        for file_path in sorted(directory.glob("*.json")):
            try:
                doc = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            metadata = dict(doc.get("metadata", {}))
            if not metadata:
                continue
            cleaned_content, keep_document, _reason = sanitize_document(
                url=str(metadata.get("source_url", "")),
                title=str(metadata.get("title", "")),
                doc_type=str(metadata.get("doc_type", "")),
                content=str(doc.get("content", "")),
            )
            if not keep_document:
                skipped_documents += 1
                continue
            metadata["storage_kind"] = storage_kind
            metadata["content_length"] = len(cleaned_content)
            documents.append(metadata)

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "total_documents": len(documents),
                "crawl_stats": crawl_stats or {},
                "documents": documents,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    data_txt_path = output_path / "data.txt"
    with data_txt_path.open("w", encoding="utf-8") as out_f:
        # Rebuild one plain-text file so later preprocessing does not depend on crawler state.
        for directory_name in ("data", "pdfs"):
            directory = output_path / directory_name
            if not directory.is_dir():
                continue
            for file_path in sorted(directory.glob("*.json")):
                try:
                    doc = json.loads(file_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                metadata = dict(doc.get("metadata", {}))
                content, keep_document, _reason = sanitize_document(
                    url=str(metadata.get("source_url", "")),
                    title=str(metadata.get("title", "")),
                    doc_type=str(metadata.get("doc_type", "")),
                    content=str(doc.get("content", "")),
                )
                content = content.strip()
                if not keep_document or not content:
                    continue
                out_f.write(content + "\n\n")
                total_written += 1

    return {
        "manifest_path": str(manifest_path),
        "data_txt_path": str(data_txt_path),
        "documents_saved": len(documents),
        "data_sources": total_written,
        "documents_skipped": skipped_documents,
    }


async def discover_urls(
    session,
    skip_sitemap: bool = False,
    skip_atoz: bool = False,
    focused_seeds_only: bool = False,
    extra_seed_urls: list[str] | None = None,
) -> list[str]:
    """Discover all starting URLs from seeds, sitemap, and the A-Z page."""
    from scraper.sitemap_parser import parse_atoz_index, parse_xml_sitemap

    all_urls = set(FOCUSED_SEED_URLS if focused_seeds_only else SEED_URLS)
    all_urls.update(extra_seed_urls or [])

    # Add known subdomain roots
    if not focused_seeds_only:
        for subdomain_url in KNOWN_SUBDOMAINS:
            all_urls.add(subdomain_url)

    # Parse XML sitemap
    if not focused_seeds_only and not skip_sitemap:
        logging.info("Parsing XML sitemap...")
        sitemap_urls = await parse_xml_sitemap(SITEMAP_URL, session)
        all_urls.update(sitemap_urls)

    # Parse A-Z index
    if not focused_seeds_only and not skip_atoz:
        logging.info("Parsing A-Z index...")
        atoz_urls = await parse_atoz_index("https://iitj.ac.in/AtoZ?lg=en", session)
        all_urls.update(atoz_urls)

    logging.info("Total seed URLs discovered: %d", len(all_urls))
    return list(all_urls)


async def main():
    args = parse_args()
    setup_logging(args.verbose)
    extra_seed_urls = load_extra_seed_urls(args)
    import aiohttp
    from scraper.crawler import IITJCrawler

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  IITJ Web Scraper v1.0")
    logger.info("  Domain: iitj.ac.in")
    logger.info("  Output: %s", os.path.abspath(args.output_dir))
    logger.info("  Max pages: %s", args.max_pages or "unlimited")
    logger.info("  Concurrency: %d", args.concurrency)
    logger.info("  Delay: %.1fs", args.delay)
    logger.info("  Extra seeds: %d", len(extra_seed_urls))
    logger.info("  Focused seeds only: %s", "yes" if args.focused_seeds_only else "no")
    logger.info("=" * 60)

    # Phase 1: build the starting URL list
    logger.info("Phase 1: Discovering URLs...")
    async with aiohttp.ClientSession() as session:
        seed_urls = await discover_urls(
            session,
            skip_sitemap=args.skip_sitemap,
            skip_atoz=args.skip_atoz,
            focused_seeds_only=args.focused_seeds_only,
            extra_seed_urls=extra_seed_urls,
        )

    # Phase 2: create the crawler and seed its queue
    logger.info("Phase 2: Initializing crawler...")
    crawler = IITJCrawler(
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        delay=args.delay,
        concurrency=args.concurrency,
        resume=args.resume,
    )
    await crawler.initialize(seed_urls)

    # Phase 3: run the actual crawl
    logger.info("Phase 3: Starting BFS crawl...")
    await crawler.crawl()

    # Phase 4: rebuild clean output files from the saved JSON documents
    logger.info("Phase 4: Rebuilding manifest and data.txt from saved documents...")
    rebuild_summary = rebuild_output_artifacts(
        args.output_dir,
        crawl_stats={
            "pages_crawled": crawler.pages_crawled,
            "pdfs_extracted": crawler.pdfs_extracted,
            "pages_failed": crawler.pages_failed,
            "dedup_stats": crawler.dedup.stats,
        },
    )

    logger.info("=" * 60)
    logger.info("  Crawl Complete!")
    logger.info("  Documents saved: %d", rebuild_summary["documents_saved"])
    logger.info("  Documents skipped in rebuild: %d", rebuild_summary["documents_skipped"])
    logger.info("  data.txt sources: %d", rebuild_summary["data_sources"])
    logger.info("  Manifest: %s", rebuild_summary["manifest_path"])
    logger.info("  Data directory: %s", os.path.join(args.output_dir, "data"))
    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        setup_logging(False)
        args = parse_args()
        logging.warning("Crawl interrupted. Rebuilding manifest/data.txt from saved documents...")
        summary = rebuild_output_artifacts(args.output_dir)
        logging.info("Recovered %d saved documents into %s", summary["documents_saved"], summary["manifest_path"])
