"""
Metadata builder: classifies documents and attaches provenance metadata.
"""

import hashlib
import re
from datetime import datetime, timezone

from config.settings import DOC_TYPE_RULES, DEFAULT_DOC_TYPE


def build_metadata(
    url: str,
    title: str = "",
    description: str = "",
    content: str = "",
    doc_type: str | None = None,
    extra: dict | None = None,
) -> dict:
    """
    Build a metadata dictionary for a scraped document.

    Args:
        url: source URL
        title: page/document title
        description: meta description
        content: the extracted text content (used for content_hash)
        doc_type: override doc type; if None, auto-classify from URL
        extra: additional metadata fields (e.g., pdf pages, file_size)

    Returns:
        Metadata dict.
    """
    if doc_type is None:
        doc_type = classify_doc_type(url)

    department = extract_department(url)

    meta = {
        "id": hashlib.sha256(url.encode()).hexdigest()[:16],
        "source_url": url,
        "title": title,
        "description": description,
        "doc_type": doc_type,
        "department": department,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "content_hash": hashlib.md5(content.encode()).hexdigest() if content else "",
        "content_length": len(content) if content else 0,
    }

    if extra:
        meta.update(extra)

    return meta


def classify_doc_type(url: str) -> str:
    """Auto-classify document type from URL patterns."""
    url_lower = url.lower()
    for pattern, dtype in DOC_TYPE_RULES:
        if re.search(pattern, url_lower):
            return dtype
    return DEFAULT_DOC_TYPE


def extract_department(url: str) -> str:
    """Extract department/school/center name from URL path."""
    # Common department path patterns
    dept_patterns = [
        (r"research\.iitj\.ac\.in/unit/department/department-of-computer-science(?:-engineering)?", "Computer Science & Engineering"),
        (r"research\.iitj\.ac\.in/unit/department/department-of-electrical-engineering", "Electrical Engineering"),
        (r"research\.iitj\.ac\.in/unit/department/department-of-mechanical-engineering", "Mechanical Engineering"),
        (r"research\.iitj\.ac\.in/unit/department/department-of-civil(?:-infrastructure-engineering)?", "Civil & Infrastructure Engineering"),
        (r"research\.iitj\.ac\.in/unit/department/department-of-chemical-engineering", "Chemical Engineering"),
        (r"research\.iitj\.ac\.in/unit/department/department-of-chemistry", "Chemistry"),
        (r"research\.iitj\.ac\.in/unit/department/department-of-physics", "Physics"),
        (r"research\.iitj\.ac\.in/unit/department/department-of-mathematics", "Mathematics"),
        (r"research\.iitj\.ac\.in/unit/department/department-of-bioscience", "Bioscience & Bioengineering"),
        (r"research\.iitj\.ac\.in/unit/department/department-of-metallurgical-materials-engineering", "Metallurgical & Materials Engineering"),
        (r"(^|//)cse\.iitj\.ac\.in|/cse/|/computer-science(?:-engineering)?/", "Computer Science & Engineering"),
        (r"(^|//)ee\.iitj\.ac\.in|/electrical(?:-engineering)?/|/ee/", "Electrical Engineering"),
        (r"(^|//)me\.iitj\.ac\.in|/mechanical(?:-engineering)?/|/me/", "Mechanical Engineering"),
        (r"(^|//)civil\.iitj\.ac\.in|/civil(?:-and-infrastructure-engineering)?/|/ce/", "Civil & Infrastructure Engineering"),
        (r"/chemical-engineering/", "Chemical Engineering"),
        (r"/chemistry/", "Chemistry"),
        (r"/physics/", "Physics"),
        (r"/mathematics/", "Mathematics"),
        (r"/bioscience(?:-bioengineering)?/", "Bioscience & Bioengineering"),
        (r"/materials-engineering/|/metallurgical/", "Metallurgical & Materials Engineering"),
        (r"/school-of-artificial-intelligence(?:-data-science)?/|/aide/", "School of AI & Data Science"),
        (r"/school-of-design/|(^|//)design\.iitj\.ac\.in", "School of Design"),
        (r"/school-of-liberal-arts/|/sola/|(^|//)sola\.iitj\.ac\.in", "School of Liberal Arts"),
        (r"/school-of-management/|/sme/|(^|//)sme\.iitj\.ac\.in", "School of Management & Entrepreneurship"),
        (r"/dh/|/digital-humanities/", "Digital Humanities"),
        (r"/office-of-academics/", "Office of Academic Affairs"),
        (r"/office-of-director/", "Office of Director"),
        (r"/office-of-students/", "Office of Students"),
        (r"/office-of-research/", "Office of R&D"),
        (r"/es/|/engineering-science/", "Engineering Science"),
        (r"/cete?/", "Center for Education Technology"),
        (r"/cetsd/", "Center for Technology & Sustainable Development"),
        (r"/crf/", "Central Research Facility"),
    ]

    url_lower = url.lower()
    for pattern, dept_name in dept_patterns:
        if re.search(pattern, url_lower):
            return dept_name

    return "Institute"
