"""HTML content extractor: turn raw HTML into cleaner text.

I try one stronger extractor first, then fall back to BeautifulSoup if needed.
"""

import logging
import re
import langid

import trafilatura
from bs4 import BeautifulSoup
from scraper.content_filters import clean_scraped_content

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
    r"RedirectToLoginPage",
    r"arrow_downward",
    r"\bA\+\s*A\s*A-\b",
    r"Home\s+People\s+Committee\s+Publications",
    r"use\s+horizontal\s+scroll\s+bar\s+table",
    r"downloaded\s+clicking\s+following\s+links",
    r"link\s+open\s+intranet",
]

CANDIDATE_NOISE_PATTERNS = [
    r"redirecttologinpage",
    r"arrow_downward",
    r"\bbooking for (?:internal|external) sample\b",
    r"\bacademic users\b",
    r"\bindustry users\b",
    r"\bcoordination committee\b",
    r"\buse horizontal scroll bar table\b",
    r"\blink open intranet\b",
    r"\bdownloaded clicking following links\b",
    r"\bclass room\b",
    r"\bcourse code\b",
    r"\bjee counseling\b",
    r"\bnational scholarship portal\b",
    r"\bhome\b",
    r"\bsitemap\b",
]


def extract_content(html: str, url: str) -> dict:
    """
    Extract clean text content from HTML.

    Returns:
        dict with keys: 'content' (str), 'title' (str), 'description' (str)
    """
    # I always return the same keys so the crawler can use the result safely.
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

    candidates: list[tuple[str, str]] = []

    # Primary extractor: trafilatura usually gives the cleanest main text.
    try:
        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            include_links=False,
            include_images=False,
            deduplicate=True,
            favor_precision=True,
            output_format="txt",
            target_language="en",
        )
        if extracted and len(extracted.strip()) > 50:
            cleaned = _clean_text(extracted)
            is_english, lang = _is_english(cleaned)
            if is_english:
                candidates.append(("trafilatura", cleaned))
            else:
                logger.debug("Trafilatura skipped non-English content (%s) for %s", lang, url)
    except Exception as e:
        logger.debug("Trafilatura failed for %s: %s", url, e)

    # Fallback extractor: simpler, but useful when trafilatura misses the content.
    try:
        content = _bs4_extract(html)
        if content and len(content.strip()) > 50:
            cleaned = _clean_text(content)
            is_english, lang = _is_english(cleaned)
            if is_english:
                candidates.append(("bs4", cleaned))
            else:
                logger.debug("BS4 skipped non-English content (%s) for %s", lang, url)
    except Exception as e:
        logger.debug("BS4 fallback failed for %s: %s", url, e)

    if candidates:
        # If both methods work, keep the candidate with the better text-quality score.
        best_source, best_text = max(candidates, key=lambda item: _candidate_score(item[1]))
        logger.debug("Selected %s extraction for %s", best_source, url)
        result["content"] = best_text

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

    # I check common "main content" containers before falling back to the whole body.
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
    for elem in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre"]):
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

    return clean_scraped_content(text)


def _is_english(text: str) -> tuple[bool, str]:
    lang, _ = langid.classify(text)
    return lang == "en", lang


def _candidate_score(text: str) -> float:
    # Prefer longer, less repetitive text with fewer boilerplate hints.
    lowered = text.lower()
    tokens = re.findall(r"[a-z]+", lowered)
    if not tokens:
        return float("-inf")

    noise_hits = sum(len(re.findall(pattern, lowered)) for pattern in CANDIDATE_NOISE_PATTERNS)
    short_lines = sum(1 for line in text.splitlines() if 0 < len(re.findall(r"[a-z]+", line.lower())) <= 2)
    unique_ratio = len(set(tokens)) / len(tokens)
    # Formula:
    # score = token_count + (unique_ratio * 100) - (noise_hits * 35) - (short_lines * 1.5)
    # Bigger score means "this looks more like useful main content".
    return len(tokens) + (unique_ratio * 100.0) - (noise_hits * 35.0) - (short_lines * 1.5)
