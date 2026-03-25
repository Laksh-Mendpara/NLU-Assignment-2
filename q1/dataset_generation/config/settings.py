"""Configuration settings for the IITJ scraper.

I kept most comments short and simple here because this file is mostly lists.
"""

# ── Seed URLs ───────────────────────────────────────────────────────────────
ACADEMIC_SEED_URLS = [
    "https://iitj.ac.in/office-of-academics/en/Academic-Regulations",
    "https://iitj.ac.in/office-of-academics/en/academics",
    "https://iitj.ac.in/office-of-academics/en/program-structure",
    "https://iitj.ac.in/office-of-academics/en/program-Structure",
    "https://iitj.ac.in/office-of-academics/en/curriculum",
    "https://iitj.ac.in/office-of-academics/en/Curriculum-for-Programs-Before-2019",
    "https://iitj.ac.in/schools/en/phd-program",
    "https://iitj.ac.in/schools/en/program-curriculum",
    "https://iitj.ac.in/schools/en/Curriculum-and-Electives",
    "https://iitj.ac.in/chemistry/en/about-research",
    "https://iitj.ac.in/chemistry/en/curriculum",
    "https://iitj.ac.in/mathematics/en/program-structure",
    "https://iitj.ac.in/mathematics/en/curriculum",
    "https://iitj.ac.in/physics/en/program-structure",
    "https://iitj.ac.in/physics/en/curriculum",
    "https://iitj.ac.in/electrical-engineering/en/research-overview",
    "https://iitj.ac.in/electrical-engineering/en/curriculum",
    "https://iitj.ac.in/cete/en/about-research",
    "https://iitj.ac.in/cete/en/phd-program",
    "https://iitj.ac.in/school-of-artificial-intelligence-data-science/en/about-research",
    "https://iitj.ac.in/centre-of-excellence-on-arts-and-digital-immersion/en/about-research",
    "https://www.iitj.ac.in/iot/en/research-areas",
    "https://www.iitj.ac.in/main/en/research-areas-removed",
    "https://iitj.ac.in/chemical-engineering/en/program-structure",
    "https://iitj.ac.in/chemical-engineering/en/curriculum",
    "https://iitj.ac.in/bioscience-bioengineering/en/program-structure",
    "https://iitj.ac.in/bioscience-bioengineering/en/curriculum",
    "https://iitj.ac.in/materials-engineering/en/program-structure",
    "https://iitj.ac.in/materials-engineering/en/curriculum",
    "https://iitj.ac.in/mechanical-engineering/en/program-structure",
    "https://iitj.ac.in/mechanical-engineering/en/curriculum",
    "https://iitj.ac.in/civil-and-infrastructure-engineering/en/program-structure",
]

RESEARCH_SEED_URLS = [
    "https://iitj.ac.in/office-of-research-development/en/office-of-research-and-development",
    "https://iitj.ac.in/office-of-research-development/en/About-Us",
    "https://iitj.ac.in/office-of-research-development/en/Collaboration",
    "https://iitj.ac.in/office-of-research-development/en/Technology",
    "https://iitj.ac.in/main/en/pmrf",
    "https://iitj.ac.in/bioscience-bioengineering/en/completed-projects",
    "https://iitj.ac.in/bioscience-bioengineering/en/laboratories",
    "https://iitj.ac.in/cete/en/About-Research",
    "https://iitj.ac.in/chemistry/en/completed-projects",
    "https://iitj.ac.in/chemistry/en/ongoing-projects",
    "https://iitj.ac.in/chemistry/en/researchrlaboratories",
    "https://iitj.ac.in/dh/en/about-research-2",
    "https://iitj.ac.in/materials-engineering/en/completed-projects",
    "https://iitj.ac.in/mathematics/en/completed-projects",
    "https://iitj.ac.in/mechanical-engineering/en/about-research",
    "https://iitj.ac.in/mechanical-engineering/en/completed-projects",
    "https://iitj.ac.in/physics/en/completed-projects",
    "https://iitj.ac.in/physics/en/research",
    "https://iitj.ac.in/physics/en/research-groups",
    "https://iitj.ac.in/computer-science-engineering/en/research",
    "https://iitj.ac.in/computer-science-engineering/en/research-area-labs",
    "https://iitj.ac.in/computer-science-engineering/en/Research-Archive",
    "https://iitj.ac.in/civil-and-infrastructure-engineering/en/research-project",
    "https://iitj.ac.in/centre-of-excellence-on-arts-and-digital-immersion/en/research-projects",
]

STUDENT_LIFE_SEED_URLS = [
    "https://iitj.ac.in/office-of-students/en/office-of-students",
    "https://iitj.ac.in/office-of-students/en/Student-Life-%40-IIT-Jodhpur",
    "https://iitj.ac.in/office-of-students/en/Campus-Life",
    "https://iitj.ac.in/office-of-students/en/campus-life",
    "https://iitj.ac.in/office-of-students/en/Facilities",
    "https://iitj.ac.in/office-of-students/en/Academics",
    "https://iitj.ac.in/bachelor-of-technology/en/hostels-facilities",
    "https://iitj.ac.in/bachelor-of-technology/en/campus-life-%40-iitj",
    "https://iitj.ac.in/Main/en/Life-%40-IIT-Jodhpur",
    "https://iitj.ac.in/main/en/Student-Life-at-IIT-Jodhpur",
]

FACULTY_SEED_URLS = [
    "https://iitj.ac.in/main/en/faculty-members",
    "https://iitj.ac.in/main/en/adjunct-faculty-members",
    "https://iitj.ac.in/main/en/visiting-faculty-members",
    "https://iitj.ac.in/chemical-engineering/en/faculty-members",
    "https://iitj.ac.in/bioscience-bioengineering/en/faculty-members",
    "https://iitj.ac.in/electrical-engineering/en/faculty-members",
    "https://iitj.ac.in/mechanical-engineering/en/faculty-members",
    "https://iitj.ac.in/chemistry/en/faculty-members",
    "https://iitj.ac.in/civil-and-infrastructure-engineering/en/faculty-members",
    "https://iitj.ac.in/mathematics/en/faculty-members",
    "https://iitj.ac.in/physics/en/faculty-members",
    "https://iitj.ac.in/materials-engineering/en/faculty-members",
]

FOCUSED_SEED_URLS = [
    "https://iitj.ac.in/office-of-academics/en/Academic-Regulations",
    "https://iitj.ac.in/office-of-academics/en/program-structure",
    "https://iitj.ac.in/schools/en/phd-program",
    *ACADEMIC_SEED_URLS,
    *RESEARCH_SEED_URLS,
    *STUDENT_LIFE_SEED_URLS,
    *FACULTY_SEED_URLS,
]

SEED_URLS = [
    "https://iitj.ac.in/main/en/iitj",
    "https://iitj.ac.in/AtoZ?lg=en",
    *ACADEMIC_SEED_URLS,
    *RESEARCH_SEED_URLS,
    *STUDENT_LIFE_SEED_URLS,
    *FACULTY_SEED_URLS,
]

SITEMAP_URL = "https://iitj.ac.in/sitemap.xml"

# ── Known Subdomains ────────────────────────────────────────────────────────
# I list these subdomains by hand so the crawler does not miss important areas.
KNOWN_SUBDOMAINS = [
    # These are the main department, school, and institute subdomains we care about.
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
    "https://research.iitj.ac.in",
    "https://cete.iitj.ac.in",
    "https://crf.iitj.ac.in",
]

# ── Domain Filter ───────────────────────────────────────────────────────────
ALLOWED_DOMAIN = "iitj.ac.in"

# ── Rate Limiting ───────────────────────────────────────────────────────────
DEFAULT_DELAY = 1.0            # wait time between requests for one worker
MAX_CONCURRENCY = 5            # how many requests we allow at the same time
REQUEST_TIMEOUT = 30           # stop waiting after this many seconds
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0       # wait grows like 2^attempt when retrying

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
# Any URL matching one of these regex patterns will be skipped.
EXCLUDED_PATTERNS = [
    r"erponline\.iitj\.ac\.in",          # ERP portal (auth-protected)
    r"/login",
    r"/signin",
    r"/Search\?",                          # search endpoints
    r"/latest-news(?:[/?]|$)",
    r"/news(?:-|and-)?newsletter(?:[/?]|$)",
    r"/seminars?-and-meetings(?:[/?]|$)",
    r"/all-announcement(?:[/?]|$)",
    r"/announcements?(?:[/?]|$)",
    r"/scholarships?(?:[/?]|$)",
    r"/(?:first-year-)?class-time-table(?:[/?]|$)",
    r"/minor-programs(?:[/?]|$)",
    r"/m/Index/",
    r"/coordination-committee(?:[/?]|$)",
    r"/booking-for-(?:external|internal)-sample(?:[/?]|$)",
    r"/(?:academic|industry)-users(?:[/?]|$)",
    r"/centre-for-continuing-education(?:[/?]|$)",
    r"/Executive-Programs(?:[/?]|$)",
    r"/meeting-mom",
    r"/instruments(?:[/?]|$)",
    r"/office-of-executive-education(?:[/?]|$)",
    r"/social-connect(?:[/?]|$)",
    r"/teacher-training(?:[/?]|$)",
    r"/vtu(?:[/?]|$)",
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
    # These non-academic areas are skipped so the corpus stays focused.
    r"library\.iitj\.ac\.in",
    r"spc\.iitj\.ac\.in",
    r"alumni\.iitj\.ac\.in",
    r"swc\.iitj\.ac\.in",
    r"intra(?:net)?\.iitj\.ac\.in",
    r"health-center",
    r"dia/",
    r"brochure",
    r"ishaan-vikaas",
    r"jawahar-navodaya",
    r"vigyan-jyoti",
    r"working-profes+ional",
    r"bootcamp",
    r"/phd-students?(?:[/?]|$)",
    r"/msc-students?(?:[/?]|$)",
    r"/mtech-phd-student(?:s)?(?:[/?]|$)",
    r"/mtech-student(?:s)?(?:[/?]|$)",
    r"/btech-student(?:s)?(?:[/?]|$)",
    r"/student-achievements?(?:[/?]|$)",
    r"/students?-presentation(?:[/?]|$)",
    r"/reporting-schedule-for-new-students",
    # NOTE: keeping office-of-academics, departments, programs, research and admissions
]

# ── Document Type Classification ────────────────────────────────────────────
# These regex rules try to guess the document type from the URL.
DOC_TYPE_RULES = [
    (r"/office-of-students/|campus-life|student-life|student-life-at-iit-jodhpur|life-%40-iit-jodhpur|hostels?-facilities|visitors?-hostel|/facilities(?:[/?]|$)|gymkhana|dining-facilities|career-services|wellbeing", "Student Life"),
    (r"research\.iitj\.ac\.in/unit/(?:department|profile|publication)/", "Research"),
    (r"about-research|research-overview|research-areas|research-area-labs|research-groups|research-projects?|ongoing-projects?|completed-projects?|/research(?:[/?]|$)|/publications?(?:[/?]|$)|/research-archive(?:[/?]|$)|/laboratories(?:[/?]|$)", "Research"),
    (r"/crf/(?:research-highlights|publications)", "Research"),
    (r"/crf/", "Facility"),
    (r"academic.*?regulation|/office-of-academics/", "Academic Regulation"),
    (r"/faculty-members(?:[/?]|$)|/main/en/faculty-members|/adjunct-faculty-members|/visiting-faculty-members", "Faculty Profile"),
    (r"curriculum|syllabus", "Course Syllabus"),
    (r"newsletter|circular", "Newsletter / Circular"),
    (r"bulletin|announcement", "Announcement"),
    (r"/research/|/crf/", "Research"),
    (r"department|/cse/|/ee/|/me/|/ce/|/civil/|/chemistry/|/physics/|/mathematics/|/bioscience/|/metallurgical/|/aide/|/sola/|/sme/|/dh/", "Department"),
    (r"program|course|curriculum-and-electives|program-curriculum", "Academic Program"),
]

DEFAULT_DOC_TYPE = "General"
