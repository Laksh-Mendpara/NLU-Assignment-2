"""
HTML content extractor: converts raw HTML to clean Markdown text.
Uses trafilatura as primary extractor with BeautifulSoup fallback.
"""

import logging
import re
import langid

import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Tags to strip during BS4 fallback
STRIP_TAGS = [
    "nav", "footer", "header", "aside", "script", "style",
    "noscript", "iframe", "form", "button", "input",
]

# CSS classes/ids commonly used for boilerplate
STRIP_SELECTORS = [
    ".navbar", ".nav", ".footer", ".sidebar", ".breadcrumb",
    ".social-share", ".social-icons", ".cookie-banner",
    "#navbar", "#footer", "#sidebar", "#header",
    ".menu", ".top-bar", ".bottom-bar",
]

# Patterns to clean from final text
NOISE_PATTERNS = [
    r"Copyright\s*©\s*\d{4}.*?(?:\n|$)",
    r"Last\s+Updated\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}.*?(?:\n|$)",
    r"All\s+Rights?\s+Reserved.*?(?:\n|$)",
    r"Designed\s+(?:and|&)\s+(?:Developed|Maintained)\s+by.*?(?:\n|$)",
    r"Visitor\s*Count\s*:?\s*\d+",
    r"Skip\s+to\s+(?:main\s+)?content",
]


def extract_content(html: str, url: str) -> dict:
    """
    Extract clean text content from HTML.

    Returns:
        dict with keys: 'content' (str), 'title' (str), 'description' (str)
    """
    result = {"content": "", "title": "", "description": ""}

    if not html:
        return result

    # Extract title with BS4 (reliable)
    try:
        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        if title_tag:
            result["title"] = title_tag.get_text(strip=True)

        # Extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            result["description"] = meta_desc["content"].strip()
    except Exception:
        pass

    # Primary: trafilatura
    try:
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            include_links=True,
            include_images=False,
            output_format="txt",
            favor_recall=True,
        )
        if extracted and len(extracted.strip()) > 50:
            cleaned = _clean_text(extracted)
            lang, _ = langid.classify(cleaned)
            if lang == 'en':
                result["content"] = cleaned
                return result
            else:
                logger.debug("Trafilatura skipped non-English content (%s) for %s", lang, url)
    except Exception as e:
        logger.debug("Trafilatura failed for %s: %s", url, e)

    # Fallback: BeautifulSoup manual extraction
    try:
        content = _bs4_extract(html)
        if content and len(content.strip()) > 50:
            cleaned = _clean_text(content)
            lang, _ = langid.classify(cleaned)
            if lang == 'en':
                result["content"] = cleaned
            else:
                logger.debug("BS4 skipped non-English content (%s) for %s", lang, url)
    except Exception as e:
        logger.debug("BS4 fallback failed for %s: %s", url, e)

    return result


def _bs4_extract(html: str) -> str:
    """
    Fallback extraction using BeautifulSoup: strip boilerplate, return text.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove boilerplate tags
    for tag_name in STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove boilerplate by CSS selector
    for selector in STRIP_SELECTORS:
        try:
            for elem in soup.select(selector):
                elem.decompose()
        except Exception:
            pass

    # Try to find main content container
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"id": "content"})
        or soup.find("div", {"class": "content"})
        or soup.find("div", {"id": "main-content"})
        or soup.find("div", {"class": "main-content"})
        or soup.body
    )

    if not main:
        return ""

    # Get text with newlines between blocks
    lines = []
    for elem in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "blockquote", "pre"]):
        text = elem.get_text(separator=" ", strip=True)
        if text:
            # Add markdown heading markers
            if elem.name.startswith("h") and len(elem.name) == 2:
                level = int(elem.name[1])
                text = "#" * level + " " + text
            elif elem.name == "li":
                text = "- " + text
            lines.append(text)

    return "\n\n".join(lines)


def _clean_text(text: str) -> str:
    """
    Remove institutional boilerplate noise from extracted text.
    """
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    return text
