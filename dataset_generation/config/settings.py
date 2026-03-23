"""
Configuration settings for IITJ web scraper.
"""

# ── Seed URLs ───────────────────────────────────────────────────────────────
SEED_URLS = [
    "https://iitj.ac.in/main/en/iitj",
    "https://iitj.ac.in/AtoZ?lg=en",
]

SITEMAP_URL = "https://iitj.ac.in/sitemap.xml"

# ── Known Subdomains ────────────────────────────────────────────────────────
# explicitly seed department/school subdomains so we don't miss them
KNOWN_SUBDOMAINS = [
    # Explicitly including department and school subdomains
    "https://cse.iitj.ac.in",
    "https://ee.iitj.ac.in",
    "https://me.iitj.ac.in",
    "https://bb.iitj.ac.in",
    "https://chem.iitj.ac.in",
    "https://maths.iitj.ac.in",
    "https://physics.iitj.ac.in",
    "https://met.iitj.ac.in",
    "https://civil.iitj.ac.in",
    "https://aide.iitj.ac.in",
    "https://sme.iitj.ac.in",
    "https://sola.iitj.ac.in",
    "https://design.iitj.ac.in",
    "https://dh.iitj.ac.in",
    # Academic/Research/Institute repositories
    "https://ir.iitj.ac.in",
    "https://cete.iitj.ac.in",
    "https://crf.iitj.ac.in",
]

# ── Domain Filter ───────────────────────────────────────────────────────────
ALLOWED_DOMAIN = "iitj.ac.in"

# ── Rate Limiting ───────────────────────────────────────────────────────────
DEFAULT_DELAY = 1.0            # seconds between requests per worker
MAX_CONCURRENCY = 5            # max simultaneous connections
REQUEST_TIMEOUT = 30           # seconds
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0       # exponential backoff base

# ── Output ──────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = "./output"
STATE_FILE = "crawler_state.json"
MANIFEST_FILE = "manifest.json"

# ── User-Agent ──────────────────────────────────────────────────────────────
USER_AGENT = (
    "IITJDataCollector/1.0 "
    "(Academic Research; NLU Assignment; "
    "+https://github.com/iitj-scraper)"
)

# ── PDF Settings ────────────────────────────────────────────────────────────
MAX_PDF_SIZE_MB = 100          # skip PDFs larger than this
PDF_DOWNLOAD_DIR = "/tmp/iitj_pdfs"

# ── Excluded URL Patterns ──────────────────────────────────────────────────
# Regex patterns — any URL matching these is skipped
EXCLUDED_PATTERNS = [
    r"erponline\.iitj\.ac\.in",          # ERP portal (auth-protected)
    r"/login",
    r"/signin",
    r"/Search\?",                          # search endpoints
    r"\.(jpg|jpeg|png|gif|svg|ico|webp|bmp)(\?|$)",   # images
    r"\.(mp4|mp3|avi|mov|wmv|flv|webm)(\?|$)",        # media
    r"\.(css|js|woff|woff2|ttf|eot)(\?|$)",           # assets
    r"\.(zip|rar|tar|gz|7z)(\?|$)",       # archives
    r"\.(xlsx|xls|pptx|ppt|docx|doc)(\?|$)",  # office docs (keep PDFs)
    r"mailto:",
    r"tel:",
    r"javascript:",
    r"#$",                                 # anchor-only links
    r"/hi/",                               # Hindi pages explicitly excluded
    # Exclude non-academic domains to narrow down to strictly requested data
    r"library\.iitj\.ac\.in",
    r"spc\.iitj\.ac\.in",
    r"alumni\.iitj\.ac\.in",
    r"swc\.iitj\.ac\.in",
    r"health-center",
    r"dia/",
    # NOTE: keeping office-of-academics, departments, programs, research and admissions
]

# ── Document Type Classification ────────────────────────────────────────────
# Maps URL path patterns to document types
DOC_TYPE_RULES = [
    (r"academic.*?regulation|/office-of-academics/", "Academic Regulation"),
    (r"/faculty(-members|-positions)?/", "Faculty Profile"),
    (r"curriculum|syllabus", "Course Syllabus"),
    (r"newsletter|circular", "Newsletter / Circular"),
    (r"bulletin|announcement", "Announcement"),
    (r"/research/|/crf/", "Research"),
    (r"department|/cse/|/ee/|/me/|/ce/|/civil/|/chemistry/|/physics/|/mathematics/|/bioscience/|/metallurgical/|/aide/|/sola/|/sme/|/dh/", "Department"),
    (r"program|course", "Academic Program"),
]

DEFAULT_DOC_TYPE = "General"
