"""Find URLs from the XML sitemap and the A-Z page."""

import logging
from xml.etree import ElementTree
from urllib.parse import urljoin

import aiohttp
from bs4 import BeautifulSoup

from config.settings import ALLOWED_DOMAIN, USER_AGENT, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


async def parse_xml_sitemap(sitemap_url: str, session: aiohttp.ClientSession) -> list[str]:
    """
    Fetch and parse an XML sitemap, returning a list of URLs.
    """
    urls = []
    try:
        async with session.get(
            sitemap_url,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            headers={"User-Agent": USER_AGENT},
        ) as resp:
            if resp.status != 200:
                logger.warning("Sitemap returned status %d: %s", resp.status, sitemap_url)
                return urls
            text = await resp.text()

        root = ElementTree.fromstring(text)

        # Some sitemap files point to more sitemap files, so I follow those too.
        for sitemap_elem in root.findall("sm:sitemap/sm:loc", NS):
            if sitemap_elem.text:
                child_urls = await parse_xml_sitemap(sitemap_elem.text.strip(), session)
                urls.extend(child_urls)

        # Normal sitemap entries directly contain page URLs.
        for url_elem in root.findall("sm:url/sm:loc", NS):
            if url_elem.text:
                url = url_elem.text.strip()
                if ALLOWED_DOMAIN in url:
                    urls.append(url)

    except ElementTree.ParseError as e:
        logger.error("Failed to parse XML sitemap %s: %s", sitemap_url, e)
    except Exception as e:
        logger.error("Error fetching sitemap %s: %s", sitemap_url, e)

    logger.info("Discovered %d URLs from sitemap: %s", len(urls), sitemap_url)
    return urls


async def parse_atoz_index(index_url: str, session: aiohttp.ClientSession) -> list[str]:
    """
    Scrape the A-Z HTML sitemap page and extract all internal links.
    """
    urls = []
    try:
        async with session.get(
            index_url,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            headers={"User-Agent": USER_AGENT},
        ) as resp:
            if resp.status != 200:
                logger.warning("A-Z index returned status %d: %s", resp.status, index_url)
                return urls
            html = await resp.text()

        # Here I simply collect all internal links from the A-Z page.
        soup = BeautifulSoup(html, "lxml")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(index_url, href)
            if ALLOWED_DOMAIN in full_url and full_url not in urls:
                urls.append(full_url)

    except Exception as e:
        logger.error("Error parsing A-Z index %s: %s", index_url, e)

    logger.info("Discovered %d URLs from A-Z index: %s", len(urls), index_url)
    return urls
