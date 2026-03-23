"""
IITJ Web Scraper — CLI Entrypoint

Orchestrates the full pipeline:
  1. Parse sitemap + A-Z index for URL discovery
  2. BFS crawl across iitj.ac.in domain
  3. Extract HTML → Markdown, PDF → Markdown
  4. Deduplicate and save with provenance metadata
  5. Generate manifest.json

Usage:
    python main.py                          # full crawl
    python main.py --max-pages 10           # test with 10 pages
    python main.py --resume                 # resume interrupted crawl
    python main.py --output-dir ./my_data   # custom output directory
"""

import argparse
import asyncio
import json
import logging
import os
import sys

# Add directory to sys path to avoid import errors when running from root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aiohttp

from config.settings import (
    DEFAULT_DELAY,
    DEFAULT_OUTPUT_DIR,
    KNOWN_SUBDOMAINS,
    MANIFEST_FILE,
    MAX_CONCURRENCY,
    SEED_URLS,
    SITEMAP_URL,
)
from scraper.crawler import IITJCrawler
from scraper.sitemap_parser import parse_xml_sitemap, parse_atoz_index


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="IITJ Web Scraper — Collect text data from iitj.ac.in",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --max-pages 5 --delay 2     # Quick test run
  python main.py                              # Full domain crawl
  python main.py --resume                     # Resume interrupted crawl
  python main.py --concurrency 10 --delay 0.5 # Faster crawl
        """,
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-pages", "-m",
        type=int, default=0,
        help="Maximum pages to crawl (0 = unlimited)",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float, default=DEFAULT_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int, default=MAX_CONCURRENCY,
        help=f"Max concurrent connections (default: {MAX_CONCURRENCY})",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from saved crawler state",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--skip-sitemap",
        action="store_true",
        help="Skip XML sitemap parsing (use only seeds + A-Z index)",
    )
    return parser.parse_args()


async def discover_urls(session: aiohttp.ClientSession, skip_sitemap: bool = False) -> list[str]:
    """
    Discover all seed URLs from sitemap, A-Z index, and known subdomains.
    """
    all_urls = set(SEED_URLS)

    # Add known subdomain roots
    for subdomain_url in KNOWN_SUBDOMAINS:
        all_urls.add(subdomain_url)

    # Parse XML sitemap
    if not skip_sitemap:
        logging.info("Parsing XML sitemap...")
        sitemap_urls = await parse_xml_sitemap(SITEMAP_URL, session)
        all_urls.update(sitemap_urls)

    # Parse A-Z index
    logging.info("Parsing A-Z index...")
    atoz_urls = await parse_atoz_index("https://iitj.ac.in/AtoZ?lg=en", session)
    all_urls.update(atoz_urls)

    logging.info("Total seed URLs discovered: %d", len(all_urls))
    return list(all_urls)


async def main():
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  IITJ Web Scraper v1.0")
    logger.info("  Domain: iitj.ac.in")
    logger.info("  Output: %s", os.path.abspath(args.output_dir))
    logger.info("  Max pages: %s", args.max_pages or "unlimited")
    logger.info("  Concurrency: %d", args.concurrency)
    logger.info("  Delay: %.1fs", args.delay)
    logger.info("=" * 60)

    # Phase 1: URL Discovery
    logger.info("Phase 1: Discovering URLs...")
    async with aiohttp.ClientSession() as session:
        seed_urls = await discover_urls(session, args.skip_sitemap)

    # Phase 2: Initialize Crawler
    logger.info("Phase 2: Initializing crawler...")
    crawler = IITJCrawler(
        output_dir=args.output_dir,
        max_pages=args.max_pages,
        delay=args.delay,
        concurrency=args.concurrency,
        resume=args.resume,
    )
    await crawler.initialize(seed_urls)

    # Phase 3: Crawl
    logger.info("Phase 3: Starting BFS crawl...")
    await crawler.crawl()

    # Phase 4: Generate manifest
    logger.info("Phase 4: Generating manifest...")
    manifest = await crawler.get_manifest()
    manifest_path = os.path.join(args.output_dir, MANIFEST_FILE)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_documents": len(manifest),
                "crawl_stats": {
                    "pages_crawled": crawler.pages_crawled,
                    "pdfs_extracted": crawler.pdfs_extracted,
                    "pages_failed": crawler.pages_failed,
                    "dedup_stats": crawler.dedup.stats,
                },
                "documents": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Phase 5: Compile data.txt
    logger.info("Phase 5: Writing aggregated data.txt...")
    data_txt_path = os.path.join(args.output_dir, "data.txt")
    data_dir = os.path.join(args.output_dir, "data")
    pdf_dir = os.path.join(args.output_dir, "pdfs")
    
    total_written = 0
    with open(data_txt_path, "w", encoding="utf-8") as out_f:
        for d_dir in [data_dir, pdf_dir]:
            if not os.path.isdir(d_dir):
                continue
            for fname in sorted(os.listdir(d_dir)):
                if fname.endswith(".json"):
                    try:
                        with open(os.path.join(d_dir, fname), "r", encoding="utf-8") as jf:
                            doc = json.load(jf)
                            content = doc.get("content", "").strip()
                            if content:
                                out_f.write(content + "\n\n")
                                total_written += 1
                    except Exception as e:
                        logger.error("Failed to read %s for data.txt: %s", fname, e)

    logger.info("=" * 60)
    logger.info("  Crawl Complete!")
    logger.info("  Documents saved: %d", len(manifest))
    logger.info("  Manifest: %s", manifest_path)
    logger.info("  Data directory: %s", os.path.join(args.output_dir, "data"))
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
