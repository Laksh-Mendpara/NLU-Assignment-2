from __future__ import annotations

import re

# These regexes help us trim noisy website content before training.
TOKEN_RE = re.compile(r"[a-z]+(?:[a-z'-]*[a-z]+)?", flags=re.IGNORECASE)
MARKDOWN_HEADING_RE = re.compile(r"^\s*(#{1,6})\s*(.*?)\s*$")
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•●▪◆■◦]|o)\s+")

EXCLUDED_DOC_TYPES = {
    "Announcement",
    "Facility",
    "General",
    "Newsletter / Circular",
}

EXCLUDED_TITLE_HINTS = (
    "academic calendar",
    "advertisement",
    "bootcamp",
    "brochure",
    "centre for continuing education",
    "contact",
    "download forms",
    "faculty positions",
    "executive education",
    "executive program",
    "executive programs",
    "events",
    "faculty list",
    "frequently asked questions",
    "ishaan vikaas",
    "internal sample",
    "latest events",
    "latest news",
    "admission to postgraduate programs",
    "list of provisionally",
    "list of selected",
    "list of shortlisted",
    "news and newsletter",
    "news announcements",
    "office of executive education",
    "our alumni",
    "past programmes",
    "phd students",
    "post doctoral fellows",
    "program delivery",
    "project staff members",
    "program pathways",
    "research highlights",
    "result",
    "results",
    "scholarship",
    "seminar",
    "social connect",
    "technical staff members",
    "teacher training",
    "vigyan jyoti",
    "working professional",
    "working professionals",
)

EXCLUDED_URL_HINTS = (
    "/acad_website/",
    "/academic-calendar",
    "/admission-postgraduate-programs",
    "/adv_",
    "/advt",
    "/bootcamp",
    "/brochure",
    "/centre-for-continuing-education",
    "/contact",
    "/download-forms",
    "/executive-program",
    "/executive-programs",
    "/events",
    "/faculty-positions",
    "/faculty-list",
    "/ishaan-vikaas",
    "/intra.iitj.ac.in",
    "/intranet.iitj.ac.in",
    "/internal-sample",
    "/jawahar-navodaya",
    "/latest-events",
    "/latest-news",
    "/list-of-provisionally",
    "/list-of-selected",
    "/list-of-shortlisted",
    "/news",
    "/newsletter",
    "/office-of-executive-education",
    "/our-alumni",
    "/phd-students",
    "/post-doctoral",
    "/program-delivery",
    "/project-staff",
    "/research-highlight",
    "/result",
    "/results",
    "/scholarship",
    "/seminar",
    "/social-connect",
    "/technical-staff",
    "/teacher-training",
    "/vigyan-jyoti",
    "/vtu/",
    "/working-profes",
)

SECTION_DROP_HEADINGS = {
    "admission timeline",
    "application fee",
    "application fees",
    "application procedure",
    "associated faculty members",
    "cancellation of admission",
    "class schedule",
    "contact us",
    "contact weeks",
    "current activities",
    "degree options",
    "eligibility",
    "eligibility criteria and admission process",
    "faculty",
    "faculty list",
    "faculty members",
    "faqs",
    "fellowships",
    "frequently asked questions",
    "important links",
    "important dates",
    "inside this issue",
    "key features of the program",
    "latest events",
    "latest news",
    "news and newsletter",
    "news announcements",
    "online instruction",
    "online instruction and assessment",
    "our alumni",
    "past activities",
    "post doctoral fellows",
    "program fee",
    "project staff members",
    "r d projects and collaborations",
    "selection procedure",
    "shortlisting procedure",
    "student achievements",
    "student presentation",
    "students",
    "technical staff members",
    "terms and conditions for award of pg degree",
    "watch our open house",
    "watch our open house 2023 bs program",
}

GENERIC_HEADINGS = {
    "associated laboratories",
    "associated research groups",
    "program structure",
    "quick access",
    "research overview",
    "research themes",
    "technology tracks",
    "topic clouds",
    "topic clouds and course mapping",
}

SECTION_DROP_HEADING_PREFIXES = (
    "admission timeline",
    "application fee",
    "application fees",
    "application procedure",
    "contact us",
    "current activities",
    "eligibility",
    "eligibility criteria",
    "faqs",
    "fellowships",
    "frequently asked questions",
    "important dates",
    "past activities",
    "program fee",
    "selection procedure",
    "shortlisting procedure",
    "terms and conditions for award",
    "watch our open house",
)

GENERIC_HEADING_PREFIXES = (
    "associated laboratories",
    "associated research groups",
    "program structure",
    "quick access",
    "research overview",
    "research themes",
    "technology tracks",
    "topic clouds",
)

LINE_DROP_PATTERNS = (
    r"^\s*\[?\s*back to index\b.*$",
    r"^\s*for any comments\s*/?\s*enquiries\s*/?\s*feedback.*wim.*$",
    r"^\s*advertisement details\s*:.*$",
    r"^\s*brochure\s*:.*$",
    r"^\s*closing date of application\s*:.*$",
    r"^\s*cut-?off/exam/interview date\s*:.*$",
    r"^\s*declaration of results\s*:.*$",
    r"^.*download file.*$",
    r"^\s*(?:detailed\s+curriculum|details\s+of\s+courses\s+offered|specializations?|regulations?\s+for\s+undergraduate\s+programs?).*download file.*$",
    r"^\s*details of the project can be found here:.*$",
    r"^\s*link for application:.*$",
    r"^\s*list of provisionally .*",
    r"^\s*list of shortlisted .*",
    r"^\s*list of selected .*",
    r"^\s*note:\s*examples of course codes.*$",
    r"^\s*please visit the following link.*$",
    r"^\s*program structure\s*:.*download file.*$",
    r"^\s*current students\s*$",
    r"^\s*graduated students\s*$",
    r"^\s*see table\s+\d+(?:\.\d+)*\b.*$",
    r"^\s*table\s+\d+(?:\.\d+)*\b.*$",
    r"^\s*the curriculum and course contents are available at.*$",
    r"^\s*this code will be given centrally.*$",
    r"^\s*view all .*",
    r"^\s*watch our open house.*$",
    r"^\s*(?:course title|course code|l-t-p|parameter|regular course|fractal course|s\.?\s*n\.?|type)\s*$",
)

GLOBAL_DROP_PATTERNS = (
    r"for any comments\s*/?\s*enquiries\s*/?\s*feedback.*?wim\s*\.*",
    r"advertisement details\s*:\s*\(download file:[^)]+\)",
    r"brochure\s*:\s*\(download file:[^)]+\)",
    r"closing date of application\s*:.*?(?:\n|$)",
    r"cut-?off/exam/interview date\s*:.*?(?:\n|$)",
    r"declaration of results\s*:.*?(?:\n|$)",
    r"program structure\s*:\s*\(download file:[^)]+\)",
)

PARAGRAPH_DROP_PHRASES = (
    "audio visual mode",
    "campus immersion",
    "fees once paid will not be refunded",
    "for any comments enquiries feedback",
    "frequently asked questions",
    "if you wish to browse research areas mapped to the technology tracks",
    "link for application",
    "please visit the following link",
    "the curriculum and course contents are available at",
    "google classroom piazza moodle",
    "inside this issue",
    "national scholarship portal",
    "off campus professionals",
    "student achievements",
    "working professionals",
)

PROMOTIONAL_HINTS = {
    "advertisement",
    "amazon",
    "application",
    "apply",
    "apple",
    "brochure",
    "bootcamp",
    "candidates",
    "closing",
    "declaration",
    "executive",
    "fee",
    "fees",
    "google",
    "immersive",
    "interview",
    "moodle",
    "piazza",
    "portal",
    "professional",
    "professionals",
    "result",
    "results",
    "salary",
    "scholarship",
    "selected",
    "shortlisted",
    "synchronous",
    "timeline",
    "waitlisted",
    "weekend",
    "weekends",
}

MIN_TOKENS_BY_DOC_TYPE = {
    "Academic Program": 120,
    "Academic Regulation": 160,
    "Course Syllabus": 180,
    "Department": 100,
    "Faculty Profile": 90,
    "Research": 100,
    "Student Life": 80,
}


def _normalize_heading(text: str) -> str:
    heading = re.sub(r"[^a-z0-9\s-]+", " ", text.lower())
    heading = re.sub(r"\s+", " ", heading).strip(" -")
    return heading


def _is_section_drop_heading(heading: str) -> bool:
    return heading in SECTION_DROP_HEADINGS or any(
        heading.startswith(prefix) for prefix in SECTION_DROP_HEADING_PREFIXES
    )


def _is_generic_heading(heading: str) -> bool:
    return heading in GENERIC_HEADINGS or any(
        heading.startswith(prefix) for prefix in GENERIC_HEADING_PREFIXES
    )


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"[•●▪◆■◦]+", "\n", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _should_drop_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped in {"-", "*", "o"}:
        return True

    lowered = _normalize_heading(stripped.lstrip("#").strip())
    if _is_section_drop_heading(lowered):
        return True

    for pattern in LINE_DROP_PATTERNS:
        if re.search(pattern, stripped, flags=re.IGNORECASE):
            return True

    token_count = len(TOKEN_RE.findall(stripped))
    if token_count <= 2 and lowered in {"rti", "recruitment", "correspondence"}:
        return True
    return False


def _should_drop_paragraph(paragraph: str) -> bool:
    lowered = _normalize_heading(paragraph)
    if not lowered:
        return True
    if any(phrase in lowered for phrase in PARAGRAPH_DROP_PHRASES):
        return True

    tokens = TOKEN_RE.findall(lowered)
    if not tokens:
        return True

    promo_hits = sum(token in PROMOTIONAL_HINTS for token in tokens)
    if len(tokens) <= 40 and promo_hits >= 3:
        return True
    if lowered.startswith("table ") or lowered.startswith("see table "):
        return True
    if lowered.startswith("title of the talk") or lowered.startswith("about the speaker"):
        return True
    return False


def clean_scraped_content(text: str) -> str:
    if not text:
        return ""

    # First clean globally, then do a second pass line-by-line and paragraph-by-paragraph.
    cleaned = _normalize_whitespace(text)
    for pattern in GLOBAL_DROP_PATTERNS:
        cleaned = re.sub(pattern, "\n", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = _normalize_whitespace(cleaned)

    lines = cleaned.splitlines()
    kept_lines: list[str] = []
    skip_section_level: int | None = None
    skip_plain_section = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if skip_plain_section:
                skip_plain_section = False
            if kept_lines and kept_lines[-1] != "":
                kept_lines.append("")
            continue

        heading_match = MARKDOWN_HEADING_RE.match(line)
        normalized_line = _normalize_heading(line)
        if heading_match:
            level = len(heading_match.group(1))
            heading = _normalize_heading(heading_match.group(2))
            if skip_section_level is not None and level <= skip_section_level:
                skip_section_level = None
            if _is_section_drop_heading(heading):
                skip_section_level = level
                continue
            if _is_generic_heading(heading):
                continue
            skip_plain_section = False
        elif _is_section_drop_heading(normalized_line) and len(TOKEN_RE.findall(normalized_line)) <= 10:
            skip_plain_section = True
            continue
        elif _is_generic_heading(normalized_line) and len(TOKEN_RE.findall(normalized_line)) <= 10:
            continue

        if skip_section_level is not None or skip_plain_section:
            continue

        if _should_drop_line(line):
            continue

        if heading_match:
            line = heading_match.group(2).strip()
        else:
            line = BULLET_PREFIX_RE.sub("", line)
            line = re.sub(r"^\s*\d+\s*[|.)-]\s*", "", line)

        kept_lines.append(line)

    paragraphs: list[str] = []
    seen_paragraphs: set[str] = set()
    for paragraph in re.split(r"\n\s*\n", "\n".join(kept_lines)):
        paragraph = _normalize_whitespace(paragraph)
        if _should_drop_paragraph(paragraph):
            continue
        dedup_key = re.sub(r"[^a-z0-9]+", " ", paragraph.lower()).strip()
        # Only deduplicate longer paragraphs so short meaningful lines are not over-pruned.
        if len(TOKEN_RE.findall(dedup_key)) >= 10:
            if dedup_key in seen_paragraphs:
                continue
            seen_paragraphs.add(dedup_key)
        paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)


def sanitize_document(
    *,
    url: str,
    title: str,
    doc_type: str,
    content: str,
) -> tuple[str, bool, str]:
    cleaned_content = clean_scraped_content(content)
    url_lower = (url or "").lower()
    title_lower = (title or "").lower()
    content_lower = cleaned_content.lower()
    content_search = _normalize_heading(cleaned_content)

    if doc_type in EXCLUDED_DOC_TYPES:
        return cleaned_content, False, f"excluded_doc_type:{doc_type}"

    if any(hint in title_lower for hint in EXCLUDED_TITLE_HINTS):
        return cleaned_content, False, "excluded_title"
    if any(hint in url_lower for hint in EXCLUDED_URL_HINTS):
        return cleaned_content, False, "excluded_url"
    if (
        "executive" in title_lower
        or "working professional" in title_lower
        or "working professional" in url_lower
        or "executive m tech" in content_search
        or "working professionals" in content_search
        or "off campus professionals" in content_search
        or "campus immersion" in content_search
    ):
        return cleaned_content, False, "executive_or_offcampus"
    if "welcome to faculty positions" in content_search or (
        "faculty positions" in content_search and "online application form" in content_search
    ):
        return cleaned_content, False, "faculty_recruitment"
    if "welcome to admission in post graduate programs" in content_search or (
        "information related to admission into these programs is posted here regularly" in content_search
    ):
        return cleaned_content, False, "admission_index"
    if "vigyan jyoti" in content_search or "jawahar navodaya" in content_search:
        return cleaned_content, False, "outreach_or_school_program"
    if re.search(r"/(?:phd|msc|mtech|btech)(?:-phd)?-students?(?:[/?]|$)", url_lower):
        return cleaned_content, False, "student_roster_page"
    if re.search(r"/students?(?:[/?]|$)", url_lower) and "office-of-students" not in url_lower:
        if "student life" not in title_lower and "campus life" not in title_lower:
            return cleaned_content, False, "student_listing_page"

    tokens = TOKEN_RE.findall(cleaned_content.lower())
    token_count = len(tokens)
    # Different page types need slightly different minimum lengths.
    min_tokens = MIN_TOKENS_BY_DOC_TYPE.get(doc_type, 80)
    if token_count < min_tokens:
        return cleaned_content, False, "too_short_after_cleaning"

    if doc_type == "Faculty Profile":
        faculty_signal_tokens = {
            "research",
            "interest",
            "interests",
            "publication",
            "project",
            "laboratory",
            "lab",
            "phd",
            "student",
            "teaching",
            "supervision",
            "group",
        }
        faculty_signal_hits = sum(token in faculty_signal_tokens for token in tokens)
        if faculty_signal_hits < 4:
            return cleaned_content, False, "faculty_directory_or_low_signal"

    unique_ratio = len(set(tokens)) / max(token_count, 1)
    promo_hits = sum(token in PROMOTIONAL_HINTS for token in tokens)
    if token_count < 220 and unique_ratio < 0.22:
        return cleaned_content, False, "low_unique_ratio"
    if token_count < 260 and promo_hits >= 12:
        return cleaned_content, False, "promotional_content"

    return cleaned_content, True, "kept"
