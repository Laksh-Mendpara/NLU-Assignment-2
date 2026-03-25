"""Preprocess scraped IITJ documents into a clean corpus for Word2Vec training."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha1
from html import unescape
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    import langid

    HAS_LANGID = True
except ImportError:
    langid = None
    HAS_LANGID = False

try:
    from wordcloud import STOPWORDS, WordCloud

    HAS_WORDCLOUD = True
except ImportError:
    STOPWORDS = set()
    WordCloud = None
    HAS_WORDCLOUD = False

try:
    import spacy

    HAS_SPACY = True
except ImportError:
    spacy = None
    HAS_SPACY = False

# These regexes are reused throughout cleaning, filtering, and tokenization.
TOKEN_PATTERN = re.compile(r"[a-z]+(?:[a-z0-9_'-]*[a-z0-9]+)?")
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n{2,}")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?;:])\s+")
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
UUID_PATTERN = re.compile(r"\b[a-f0-9]{8,}(?:-[a-f0-9]{4,})+\b", flags=re.IGNORECASE)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]*)\)", flags=re.DOTALL)
PAGE_MARKER_PATTERN = re.compile(r"^\s*(?:page\s+\d+|\d+\s*\|\s*page|\d+)\s*$", flags=re.IGNORECASE)
MONTH_SLUG_PATTERN = re.compile(r".*_(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)(?:_\d{2,4})?$")
COURSE_CODE_PATTERN = re.compile(r"^[a-z]{2,4}\d{3}[a-z]?$")
ROMAN_NUMERAL_PATTERN = re.compile(r"^(?:ii|iii|iv|v|vi|vii|viii|ix|x)$")
SPACY_MODEL_NAME = "en_core_web_sm"

PHRASE_REPLACEMENTS = (
    (r"\bdoctor of philosophy\b", "phd"),
    (r"\bbachelor of technology\b", "btech"),
    (r"\bmaster of technology\b", "mtech"),
    (r"\bmaster of science\b", "msc"),
    (r"\bbachelor of science\b", "bsc"),
    (r"\bunder[\s-]*graduate\b", "undergraduate"),
    (r"\bpost[\s-]*graduate\b", "postgraduate"),
    (r"\bcomputer science(?:\s+and\s+|[\s&]+)?engineering\b", "computer_science_engineering"),
    (r"\belectrical engineering\b", "electrical_engineering"),
    (r"\bmechanical engineering\b", "mechanical_engineering"),
    (r"\bcivil(?:\s+and\s+|[\s&]+)?infrastructure engineering\b", "civil_infrastructure_engineering"),
    (r"\bchemical engineering\b", "chemical_engineering"),
    (r"\bmaterials engineering\b", "materials_engineering"),
    (r"\bbioscience(?:\s+and\s+|[\s&]+)?bioengineering\b", "bioscience_bioengineering"),
    (r"\bdata(?:\s+and)? computational sciences?\b", "data_computational_sciences"),
    (r"\bai(?:\s+and\s+|[\s&]+)?data science\b", "ai_data_science"),
    (r"\bdata science\b", "data_science"),
    (r"\bartificial intelligence\b", "artificial_intelligence"),
    (r"\barts(?:\s+and\s+|[\s&]+)?digital immersion\b", "arts_digital_immersion"),
    (r"\bdigital humanities\b", "digital_humanities"),
    (r"\bcyber[\s-]*physical systems\b", "cyber_physical_systems"),
    (r"\binternet of things\b", "internet_of_things"),
    (r"\bmachine learning\b", "machine_learning"),
    (r"\bdeep learning\b", "deep_learning"),
    (r"\bteaching[\s-]*learning\b", "teaching_learning"),
    (r"\bmake[\s-]*up\b", "makeup"),
    (r"\bself[\s-]*sponsored\b", "self_sponsored"),
    (r"\bindustry[\s-]*sponsored\b", "industry_sponsored"),
    (r"\bfull[\s-]*time\b", "full_time"),
    (r"\bpart[\s-]*time\b", "part_time"),
    (r"\bacademic regulations?\b", "academic_regulation"),
    (r"\boffice of academic affairs\b", "office_of_academic_affairs"),
    (r"\boffice of academics affairs\b", "office_of_academic_affairs"),
    (r"\boffice of students\b", "office_of_students"),
    (r"\bstudent life\b", "student_life"),
    (r"\bstudents gymkhana\b", "student_gymkhana"),
    (r"\bresearch and development\b", "research_development"),
    (r"\bschool of management(?:\s+and\s+|[\s&]+)?entrepreneurship\b", "school_of_management_entrepreneurship"),
    (r"\bm[\s./-]*s\s*\(\s*by research\s*\)", "ms_research"),
    (r"\bm[\s./-]*s\s+by research\b", "ms_research"),
    (r"\bmasters?\s+by research\b", "ms_research"),
    (r"\bph[\s./-]*d\b", "phd"),
    (r"\bb[\s./-]*tech\b", "btech"),
    (r"\bm[\s./-]*tech\b", "mtech"),
    (r"\bb[\s./-]*sc\b", "bsc"),
    (r"\bm[\s./-]*sc\b", "msc"),
    (r"\bp[\s./-]*g\b", "pg"),
    (r"\bu[\s./-]*g\b", "ug"),
)

TOKEN_NORMALIZATION = {
    "datum": "data",
    "programmes": "program",
    "programme": "program",
    "programs": "program",
    "courses": "course",
    "students": "student",
    "faculties": "faculty",
    "departments": "department",
    "doctoral": "phd",
    "doctorate": "phd",
    "examination": "exam",
    "examinations": "exam",
    "exams": "exam",
    "regulations": "regulation",
    "credits": "credit",
    "degrees": "degree",
    "theses": "thesis",
    "publications": "publication",
    "projects": "project",
    "papers": "paper",
    "journals": "journal",
    "semesters": "semester",
    "admissions": "admission",
    "candidates": "candidate",
    "mtech-phd": "mtech_phd",
}
COMPOUND_TOKEN_WHITELIST = {
    "academic_regulation",
    "ai_data_science",
    "artificial_intelligence",
    "arts_digital_immersion",
    "assistantship_fellowship",
    "bioscience_bioengineering",
    "chemical_engineering",
    "civil_infrastructure_engineering",
    "comprehensive_exam",
    "computer_science_engineering",
    "cyber_physical_systems",
    "data_computational_sciences",
    "data_science",
    "deep_learning",
    "digital_humanities",
    "electrical_engineering",
    "faculty_advisor",
    "full_time",
    "industry_sponsored",
    "internet_of_things",
    "machine_learning",
    "materials_engineering",
    "mechanical_engineering",
    "ms_research",
    "mtech_phd",
    "office_of_academic_affairs",
    "office_of_students",
    "part_time",
    "phd_comprehensive",
    "research_project",
    "school_of_management_entrepreneurship",
    "self_sponsored",
    "teaching_learning",
    "undergraduate_program",
    "viva_voce_exam",
}

NOISE_TOKENS = {
    "pageimages",
    "adi",
    "gallery",
    "download",
    "javascript",
    "void",
    "dslid",
    "redirecttologinpage",
    "updated",
    "email",
    "wim",
    "pm",
    "am",
    "kb",
    "mb",
    "approx",
    "click",
    "latest",
    "newsletter",
    "redirect",
    "redirecttologinpage",
    "page",
    "image",
    "images",
    "ay",
    "iitj",
    "clickhere",
    "arrow_downward",
    "sitemap",
    "home",
    "call",
    "place",
    "school",
    "deptt",
    "here",
    "view",
    "online",
    "th",
    "idrp",
    "idrps",
    "dsl",
    "gbps",
}

NOISE_PHRASES = (
    "for any comments enquiries feedback please email the wim",
    "last updated",
    "download file",
    "javascript void",
    "pageimages gallery",
    "dsl dslid",
    "the curriculum and course contents are available at",
    "please visit the following link",
    "link for application",
    "click to know more",
    "view all latest news on old website",
    "please view all news on old website",
)

EXCLUDED_TITLE_HINTS = (
    "centre for continuing education",
    "contact",
    "people",
    "download forms",
    "circulars",
    "frequently asked questions",
    "ishaan vikaas",
    "old regulations",
    "research highlights",
    "phd students",
    "executive program",
    "executive programs",
    "office of executive education",
    "past programmes",
    "program delivery",
    "social connect",
    "working professionals",
    "teacher training",
    "upcoming programmes",
    "vigyan jyoti",
    "vigyan-jyoti",
    "executive education",
    "news and newsletter",
    "news & newsletter",
    "latest news",
    "latest events",
    "seminars and meetings",
    "seminar series",
    "list of provisionally",
    "list of selected",
    "list of shortlisted",
    "result",
    "results",
    "interview schedule",
    "reporting details",
    "visit of",
    "coordination committee",
    "academic users",
    "industry users",
    "booking for external sample",
    "booking for internal sample",
    "meeting moms",
    "cut off",
    "cut-off",
    "office of admission",
    "study in india",
    "direct admission",
    "provisionally admitted",
    "scholarship",
    "brochure",
    "bootcamp",
    "working professional",
    "student list",
    "student details",
    "student achievements",
    "students presentation",
    "student presentation",
    "class time table",
    "class timetable",
    "technical staff",
)

EXCLUDED_URL_HINTS = (
    "intranet.iitj.ac.in",
    "intra.iitj.ac.in",
    "/acad_website/",
    "/centre-for-continuing-education",
    "/contact",
    "/people",
    "/ishaan-vikaas",
    "/research-highlights",
    "/phd-students",
    "/executive-programs",
    "/office-of-executive-education",
    "/programs-for-working-professionals",
    "/program-delivery",
    "/social-connect",
    "/teacher-training",
    "/upcoming-programmes",
    "/vtu/",
    "/jawahar-navodaya",
    "/vigyan-jyoti-program",
    "/news",
    "/newsletter",
    "/latest-news",
    "/latest-events",
    "/seminars",
    "/seminars-and-meetings",
    "/list-of-provisionally",
    "/list-of-selected",
    "/list-of-shortlisted",
    "/result",
    "/results",
    "/reporting-details",
    "/interview-schedule",
    "/coordination-committee",
    "/booking-for-external-sample",
    "/booking-for-internal-sample",
    "/academic-users",
    "/industry-users",
    "/meeting-mom",
    "/cut-off",
    "/office-of-admission",
    "/study-in-india",
    "/scholarships",
    "/minor-programs",
    "/adv_",
    "/advt",
    "/brochure",
    "/bootcamp",
    "/working-profes",
    "/time-table",
    "/timetable",
    "/technical-staff",
)

FORCE_INCLUDE_HINTS = (
    "research project",
    "research overview",
    "research areas",
    "about research",
    "ongoing projects",
    "completed projects",
    "publication",
    "program structure",
    "curriculum",
    "program pathway",
    "academic regulation",
    "postgraduate program",
    "undergraduate program",
    "phd program",
)

DOC_TYPE_KEEP = {
    "Research",
    "Academic Regulation",
    "Academic Program",
    "Course Syllabus",
    "Faculty Profile",
    "Department",
    "Student Life",
}

DOC_TYPE_MIN_CONTENT = {
    "Research": 180,
    "Academic Regulation": 300,
    "Academic Program": 250,
    "Course Syllabus": 400,
    "Faculty Profile": 150,
    "Department": 250,
    "Student Life": 140,
}

ANNOUNCEMENT_HINT_TOKENS = {
    "admission",
    "shortlisted",
    "shortlisting",
    "selected",
    "candidate",
    "candidates",
    "reporting",
    "merit",
    "semester",
    "round",
    "interview",
    "result",
    "results",
    "exam",
    "announcement",
    "application",
    "applications",
    "advertisement",
    "brochure",
    "submission",
    "shortlisted",
    "selected",
    "syllabus",
    "report",
    "details",
    "written",
    "test",
    "deadline",
    "schedule",
    "download",
    "fee",
    "fees",
}

TITLE_NOISE_TOKENS = {"iit", "iitj", "jodhpur", "indian", "institute", "technology"}
SHORT_LINE_NOISE_TOKENS = {
    "home",
    "people",
    "committee",
    "publications",
    "contact",
    "instruments",
    "sitemap",
    "hindi",
    "place",
    "school",
    "email",
    "call",
    "academic",
    "industry",
    "users",
    "booking",
    "system",
}
ROSTER_LINE_HINTS = (
    "back to index",
    "course title",
    "course code",
    "name of applicant",
    "name of student",
    "student id",
    "student details",
    "cut-off marks",
    "cut off marks",
    "remarks if any",
    "academic year batch",
    "list of selected",
    "list of shortlisted",
    "sl student id",
    "course instructor slot class room lab section",
    "course credit ltp course instructor",
    "examples of course codes",
    "link for application",
    "the curriculum and course contents are available at",
    "use horizontal scroll bar table",
    "link open intranet",
    "downloaded clicking following links",
)
PARAGRAPH_BOILERPLATE_PHRASES = (
    "announcement of shortlisted candidates",
    "application processing fee",
    "commencement of online application process",
    "downloaded clicking following links",
    "fees once paid will not be refunded",
    "for any query with respect to the online application",
    "link open intranet",
    "link for application",
    "please visit the following link",
    "the curriculum and course contents are available at",
    "use horizontal scroll bar table",
    "course instructor slot class room",
    "course code course credit",
    "faculty advisors year section",
    "section group allocation",
    "national scholarship portal",
    "google classroom piazza moodle",
    "off campus professionals",
    "campus immersion",
    "audio visual mode",
    "working professionals",
)
SYLLABUS_METADATA_TOKENS = {
    "classroom",
    "classrooms",
    "clicking",
    "content",
    "contents",
    "course_contents",
    "downloaded",
    "faculty_advisor",
    "faculty_advisors",
    "instructor",
    "instructors",
    "intranet",
    "isbn",
    "learning_outcomes",
    "ltp",
    "objective",
    "objectives",
    "portal",
    "prerequisite",
    "prerequisites",
    "publisher",
    "publishers",
    "reference",
    "references",
    "scholarship",
    "slot",
    "textbook",
    "textbooks",
}
REFERENCE_SECTION_TOKENS = {
    "book",
    "books",
    "cambridge",
    "edition",
    "elsevier",
    "francis",
    "mcgraw",
    "oxford",
    "pearson",
    "press",
    "publisher",
    "publishers",
    "routledge",
    "springer",
    "taylor",
    "vol",
    "wiley",
}
PROMOTIONAL_SECTION_TOKENS = {
    "abroad",
    "air",
    "amazon",
    "apple",
    "application",
    "applications",
    "admission",
    "admissions",
    "banking",
    "cgpa",
    "counseling",
    "ecommerce",
    "executive",
    "fee",
    "fees",
    "google",
    "jee",
    "placement",
    "placements",
    "professional",
    "professionals",
    "query",
    "queries",
    "rank",
    "startup",
    "startups",
    "timeline",
    "weekend",
    "weekends",
}
DEGREE_HINT_TOKENS = {
    "phd",
    "mtech",
    "btech",
    "msc",
    "bsc",
    "ug",
    "pg",
    "undergraduate",
    "postgraduate",
    "mtech_phd",
    "ms_research",
}
ROSTER_MARKER_TOKENS = {
    "student",
    "student_id",
    "applicant",
    "applicants",
    "batch",
    "remarks",
    "regular",
    "part_time",
    "full_time",
    "name",
    "cutoff",
    "cut_off",
    "interview",
    "admission",
    "office",
    "gen",
    "obc",
    "ews",
    "pwd",
    "slot",
    "category",
    "list",
}
CONTENT_ANCHOR_TOKENS = {
    "campus",
    "research",
    "program",
    "course",
    "curriculum",
    "credit",
    "faculty",
    "department",
    "lab",
    "laboratory",
    "exam",
    "project",
    "publication",
    "seminar",
    "innovation",
    "gymkhana",
    "student_life",
    "hostel",
    "housing",
    "mess",
    "residence",
    "residential",
    "technology",
    "thesis",
    "dissertation",
    "wellbeing",
}
PHRASE_SEED_TOKENS = CONTENT_ANCHOR_TOKENS | DEGREE_HINT_TOKENS | {
    "academic",
    "admission",
    "advisor",
    "assistantship",
    "booklet",
    "campus",
    "candidacy",
    "committee",
    "credit",
    "curriculum",
    "eligible",
    "facility",
    "learning",
    "outcomes",
    "proposal",
    "registration",
    "room",
    "semester",
    "student",
    "thrust",
    "warden",
}
TITLE_ANCHOR_TOKENS = PHRASE_SEED_TOKENS | {
    "office_of_academic_affairs",
    "academic_regulation",
}
PHRASE_BLOCKLIST_TOKENS = {
    "assistant",
    "associate",
    "bb",
    "cat",
    "chair",
    "category",
    "dean",
    "dcs",
    "director",
    "dr",
    "es",
    "idrp",
    "idrps",
    "indian",
    "institute",
    "jodhpur",
    "july",
    "ltp",
    "march",
    "members",
    "mr",
    "mrs",
    "ms",
    "nh",
    "number",
    "oa",
    "ope",
    "phone",
    "professor",
    "sir",
    "sports",
    "technology_jodhpur",
}
PHRASE_MIN_COUNT = 8
PHRASE_MIN_UNIGRAM_COUNT = 10
PHRASE_MIN_SCORE = 8.0
PHRASE_MAX_VOCAB = 200
PROTECTED_TRAINING_TOKENS = PHRASE_SEED_TOKENS | {
    "btech",
    "course",
    "credit",
    "department",
    "exam",
    "faculty",
    "hostel",
    "mtech",
    "phd",
    "program",
    "room",
    "research",
    "student",
    "thesis",
    "undergraduate",
    "postgraduate",
}
FORCED_DROP_TRAINING_TOKENS = set(ENGLISH_STOP_WORDS) | {
    "admission",
    "admitted",
    "applicant",
    "applicants",
    "approval",
    "approved",
    "attendance",
    "ability",
    "advanced",
    "analysis",
    "applications",
    "assistantships",
    "barc",
    "bhabha",
    "book",
    "books",
    "calendar",
    "cambridge",
    "certificate",
    "certificates",
    "cgpa",
    "clicking",
    "continuing",
    "content",
    "contents",
    "counseling",
    "dates",
    "december",
    "deficiency",
    "deficient",
    "downloaded",
    "educator",
    "edition",
    "elsevier",
    "engineering",
    "exempted",
    "expected",
    "followed",
    "gandhinagar",
    "granted",
    "guidance",
    "instructor",
    "intranet",
    "institute",
    "isbn",
    "issued",
    "introduction",
    "iit",
    "india",
    "indian",
    "indo",
    "invited",
    "invites",
    "january",
    "jodhpur",
    "jee",
    "june",
    "knowledge",
    "learn",
    "learner",
    "letter",
    "letters",
    "lecture",
    "lectures",
    "learning_outcomes",
    "ls",
    "ltp",
    "mcgraw",
    "methods",
    "mid",
    "minutes",
    "moe",
    "months",
    "nd",
    "oe",
    "number",
    "objective",
    "objectives",
    "offered",
    "offer",
    "offers",
    "open",
    "orientation",
    "oxford",
    "paid",
    "pc",
    "pe",
    "pearson",
    "permission",
    "pp",
    "prescribed",
    "prerequisite",
    "press",
    "process",
    "processes",
    "publisher",
    "publishers",
    "psychiatry",
    "reference",
    "registered",
    "registration",
    "references",
    "routledge",
    "sc",
    "se",
    "seek",
    "separate",
    "semester_wise",
    "shall",
    "slot",
    "springer",
    "st",
    "specified",
    "system",
    "systems",
    "taylor",
    "textbook",
    "textbooks",
    "technology",
    "teaching",
    "teacher",
    "touch",
    "transcript",
    "title",
    "type",
    "understanding",
    "university",
    "various",
    "vacation",
    "vol",
    "warning",
    "wf",
    "wiley",
    "withdrawal",
    "program_pdf",
}
PEDAGOGY_OVERLOAD_TOKENS = {
    "assessment",
    "classroom",
    "curriculum",
    "educator",
    "gamification",
    "learn",
    "learner",
    "pedagogy",
    "teacher",
    "teaching",
    "tutoring",
}
TRAINING_DROP_FREQ_RATIO = 0.004
TRAINING_DROP_DOC_RATIO = 0.08
CRF_KEEP_TITLE_HINTS = ("research highlights", "publications")
CRF_EQUIPMENT_TITLE_HINTS = (
    "central research facility",
    "analyzer",
    "spectrometer",
    "microscope",
    "chromatography",
    "diffractometer",
    "calorimetry",
    "magnetometer",
    "rheometer",
    "printer",
    "surface area",
    "xrd",
    "nmr",
    "ftir",
    "facs",
    "dls",
    "trpl",
    "sample preparation",
)
MIN_SEGMENT_TOKENS = 15
TARGET_SEGMENT_TOKENS = 48
MAX_SEGMENT_TOKENS = 96
SEGMENT_OVERLAP_TOKENS = 10
MAX_SEGMENTS_PER_DOCUMENT = 120
DOC_TYPE_SEGMENT_LIMITS = {
    "Research": 64,
    "Academic Regulation": 14,
    "Academic Program": 24,
    "Course Syllabus": 10,
    "Faculty Profile": 12,
    "Department": 32,
    "Student Life": 36,
}
MIN_SEGMENT_CHARS = 35


@dataclass
class CorpusDocument:
    metadata: dict
    content: str
    source_path: Path


@dataclass
class CandidateSegment:
    metadata: dict
    segment_index: int
    tokens: list[str]


def parse_args() -> argparse.Namespace:
    # These are the fixed files used in the assignment folder.
    return argparse.Namespace(
        output_dir="output",
        manifest_file="output/manifest.json",
        input_file="output/data.txt",
        processed_corpus="output/processed_corpus.txt",
        segments_jsonl="output/processed_segments.jsonl",
        stats_file="output/dataset_stats.json",
        mapping_file="output/normalization_map.json",
        wordcloud_img="output/wordcloud.png",
        disable_langid=False,
    )


def normalize_domain_phrases(text: str) -> str:
    for pattern, replacement in PHRASE_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_token_basic(token: str) -> str:
    token = token.lower().strip().strip("'")
    if not token:
        return ""
    return TOKEN_NORMALIZATION.get(token, token)


def expand_mapped_token(mapped: str) -> list[str]:
    if not mapped:
        return []
    if "_" in mapped and mapped not in COMPOUND_TOKEN_WHITELIST:
        expanded: list[str] = []
        for part in mapped.split("_"):
            part = normalize_token_basic(part)
            if not part:
                continue
            if "_" in part and part not in COMPOUND_TOKEN_WHITELIST:
                expanded.extend(subpart for subpart in part.split("_") if subpart)
            else:
                expanded.append(part)
        return expanded
    if "-" in mapped and mapped not in COMPOUND_TOKEN_WHITELIST:
        expanded: list[str] = []
        for part in mapped.split("-"):
            part = normalize_token_basic(part)
            if not part:
                continue
            if "-" in part and part not in COMPOUND_TOKEN_WHITELIST:
                expanded.extend(subpart for subpart in part.split("-") if subpart)
            else:
                expanded.append(part)
        return expanded
    return [mapped]


@lru_cache(maxsize=1)
def get_spacy_nlp():
    if not HAS_SPACY:
        return None

    try:
        return spacy.load(SPACY_MODEL_NAME, disable=["textcat"])
    except Exception:
        try:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        except Exception:
            return None


@lru_cache(maxsize=20000)
def lemmatize_simple_token(token: str) -> str:
    token = token.lower().strip()
    if not token:
        return ""

    nlp = get_spacy_nlp()
    if nlp is None:
        return normalize_token_basic(token)

    try:
        doc = nlp(token)
        if doc:
            lemma = doc[0].lemma_.lower().strip()
            if lemma and lemma != "-pron-":
                token = lemma
    except Exception:
        pass

    return normalize_token_basic(token)


def normalize_token(token: str) -> str:
    token = token.lower().strip()
    if not token:
        return ""
    if "_" in token and token in COMPOUND_TOKEN_WHITELIST:
        return normalize_token_basic(token)
    if "-" in token and token in COMPOUND_TOKEN_WHITELIST:
        return normalize_token_basic(token)
    return lemmatize_simple_token(token)


def expand_normalized_token(token: str) -> list[str]:
    return expand_mapped_token(normalize_token(token))


def is_noise_token(token: str) -> bool:
    if token in NOISE_TOKENS:
        return True
    if token in {"s", "'s"}:
        return True
    if len(token) <= 1:
        return True
    if token.isdigit():
        return True
    if COURSE_CODE_PATTERN.fullmatch(token):
        return True
    if ROMAN_NUMERAL_PATTERN.fullmatch(token):
        return True
    if MONTH_SLUG_PATTERN.fullmatch(token):
        return True
    if token.startswith("dsl") or token.startswith("pageimage"):
        return True
    parts = token.split("_")
    if len(parts) >= 2 and any(part in {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"} for part in parts):
        return True
    if len(parts) >= 2 and any(part.isdigit() and len(part) >= 2 for part in parts):
        return True
    if any(char.isdigit() for char in token) and len(token) >= 6:
        return True
    if token.count("-") >= 2:
        return True
    if re.fullmatch(r"[a-f0-9_/-]{8,}", token):
        return True
    return False


def is_roster_like_segment(tokens: list[str]) -> bool:
    if not tokens:
        return False

    degree_count = sum(token in DEGREE_HINT_TOKENS for token in tokens)
    roster_marker_count = sum(token in ROSTER_MARKER_TOKENS for token in tokens)
    course_code_count = sum(COURSE_CODE_PATTERN.fullmatch(token) is not None for token in tokens)
    content_anchor_count = sum(token in CONTENT_ANCHOR_TOKENS for token in tokens)
    unique_ratio = len(set(tokens)) / len(tokens)

    if course_code_count >= 2:
        return True
    if degree_count >= 4 and roster_marker_count >= 1:
        return True
    if degree_count >= 5 and content_anchor_count <= 2:
        return True
    if degree_count >= 4 and unique_ratio >= 0.72 and content_anchor_count <= 4:
        return True
    return False


def is_english_text(text: str) -> bool:
    if not HAS_LANGID:
        return True
    sample = " ".join(text.split())[:2000]
    if len(sample) < 80:
        return True
    language, _score = langid.classify(sample)
    return language == "en"


def load_documents_from_manifest(manifest_path: Path, output_dir: Path) -> list[CorpusDocument]:
    from dataset_generation.scraper.content_filters import sanitize_document

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents: list[CorpusDocument] = []
    for entry in payload.get("documents", []):
        doc_id = entry["id"]
        candidates = [output_dir / "data" / f"{doc_id}.json", output_dir / "pdfs" / f"{doc_id}.json"]
        source_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if source_path is None:
            continue
        raw_doc = json.loads(source_path.read_text(encoding="utf-8"))
        metadata = raw_doc.get("metadata", {})
        merged_metadata = {**metadata, **entry}
        content, keep_document, _reason = sanitize_document(
            url=str(merged_metadata.get("source_url", "")),
            title=str(merged_metadata.get("title", "")),
            doc_type=str(merged_metadata.get("doc_type", "")),
            content=str(raw_doc.get("content", "")),
        )
        if not keep_document:
            continue
        documents.append(CorpusDocument(metadata=merged_metadata, content=content, source_path=source_path))
    return documents


def load_documents_from_directories(output_dir: Path) -> list[CorpusDocument]:
    from dataset_generation.scraper.content_filters import sanitize_document

    documents: list[CorpusDocument] = []
    for subdir in ("data", "pdfs"):
        directory = output_dir / subdir
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            raw_doc = json.loads(path.read_text(encoding="utf-8"))
            metadata = raw_doc.get("metadata", {})
            content, keep_document, _reason = sanitize_document(
                url=str(metadata.get("source_url", "")),
                title=str(metadata.get("title", "")),
                doc_type=str(metadata.get("doc_type", "")),
                content=str(raw_doc.get("content", "")),
            )
            if not keep_document:
                continue
            documents.append(
                CorpusDocument(
                    metadata=metadata,
                    content=content,
                    source_path=path,
                )
            )
    return documents


def merge_documents(*document_groups: list[CorpusDocument]) -> list[CorpusDocument]:
    merged: dict[str, CorpusDocument] = {}
    for documents in document_groups:
        for document in documents:
            metadata = document.metadata or {}
            key = str(metadata.get("id") or metadata.get("source_url") or document.source_path)
            existing = merged.get(key)
            if existing is None or len(document.content) > len(existing.content):
                merged[key] = document
    return list(merged.values())


def collect_corpus_documents(output_dir: Path, manifest_path: Path, fallback_input_file: Path) -> tuple[list[CorpusDocument], str]:
    manifest_documents: list[CorpusDocument] = []
    directory_documents: list[CorpusDocument] = []

    if manifest_path.exists():
        manifest_documents = load_documents_from_manifest(manifest_path, output_dir)

    directory_documents = load_documents_from_directories(output_dir)
    combined_documents = merge_documents(manifest_documents, directory_documents)
    if combined_documents:
        if manifest_documents and directory_documents:
            return combined_documents, "manifest+json_directories"
        if manifest_documents:
            return combined_documents, "manifest"
        return combined_documents, "json_directories"

    if fallback_input_file.exists():
        fallback_docs = [
            CorpusDocument(
                metadata={
                    "id": f"fallback-{index:06d}",
                    "title": "",
                    "doc_type": "Fallback",
                    "department": "Institute",
                    "source_url": str(fallback_input_file),
                },
                content=document,
                source_path=fallback_input_file,
            )
            for index, document in enumerate(fallback_input_file.read_text(encoding="utf-8").split("\n\n"))
            if document.strip()
        ]
        return fallback_docs, "fallback_data_txt"

    return [], "missing"


def is_recruitment_page(document: CorpusDocument) -> bool:
    title = str(document.metadata.get("title", "")).lower()
    content = (document.content or "").lower()
    return (
        "faculty members" in title
        and "applications are invited" in content
        and ("professor" in content or "positions in department" in content)
    ) or "faculty positions" in title or (
        any(hint in title for hint in ("project staff members", "pdf positions", "post-doctoral", "post doctoral"))
        and ("apply" in content or "applications are invited" in content)
    )


def is_crf_facility_page(document: CorpusDocument) -> bool:
    title = str(document.metadata.get("title", "")).lower()
    source_url = str(document.metadata.get("source_url", "")).lower()
    if "/crf/" not in source_url:
        return False
    if any(hint in title for hint in CRF_KEEP_TITLE_HINTS) or any(
        hint in source_url for hint in ("/research-highlights", "/publications")
    ):
        return False
    if any(hint in title for hint in CRF_EQUIPMENT_TITLE_HINTS):
        return True
    return any(
        hint in source_url
        for hint in (
            "/booking-for-",
            "/academic-users",
            "/industry-users",
            "/coordination-committee",
            "/contact",
            "/instruments",
            "/meeting-mom",
        )
    )


def should_keep_document(document: CorpusDocument, use_langid: bool) -> tuple[bool, str]:
    metadata = document.metadata
    title = str(metadata.get("title", "")).lower()
    source_url = str(metadata.get("source_url", "")).lower()
    doc_type = str(metadata.get("doc_type", ""))
    content = document.content or ""
    content_length = len(content.strip())
    force_include = any(hint in title or hint in source_url for hint in FORCE_INCLUDE_HINTS)
    content_tokens = [normalize_token_basic(token) for token in TOKEN_PATTERN.findall(content.lower())]

    if use_langid and not is_english_text(" ".join([title, content[:1500]])):
        return False, "non_english"

    if is_recruitment_page(document):
        return False, "excluded_recruitment_page"

    if is_crf_facility_page(document):
        return False, "excluded_facility_page"

    if doc_type == "Facility":
        return False, "excluded_doc_type:Facility"

    if doc_type not in DOC_TYPE_KEEP and not force_include:
        return False, f"excluded_doc_type:{doc_type or 'unknown'}"

    if any(hint in title for hint in EXCLUDED_TITLE_HINTS) or any(hint in source_url for hint in EXCLUDED_URL_HINTS):
        if not force_include:
            return False, "excluded_by_title_or_url"

    minimum_length = DOC_TYPE_MIN_CONTENT.get(doc_type, 250)
    if content_length < minimum_length:
        return False, f"too_short:{doc_type or 'unknown'}"

    if "/cete/" in source_url or "future education" in title:
        pedagogy_hits = sum(token in PEDAGOGY_OVERLOAD_TOKENS for token in content_tokens)
        research_hits = sum(token in {"research", "project", "lab", "publication", "phd"} for token in content_tokens)
        if pedagogy_hits >= 12 and (research_hits <= 6 or "/cete/" in source_url):
            return False, "pedagogy_heavy_page"

    return True, "kept"


def title_to_tokens(title: str) -> list[str]:
    normalized = clean_text(title).lower()
    tokens = []
    for token in TOKEN_PATTERN.findall(normalized):
        for mapped in expand_normalized_token(token):
            if mapped in TITLE_NOISE_TOKENS or is_noise_token(mapped):
                continue
            tokens.append(mapped)
    return tokens


def should_drop_line(line: str) -> bool:
    lowered = line.strip().lower()
    if not lowered:
        return True
    if PAGE_MARKER_PATTERN.fullmatch(lowered):
        return True
    if lowered.startswith("table ") or lowered.startswith("see table "):
        return True
    if any(phrase in lowered for phrase in NOISE_PHRASES):
        return True
    if any(
        marker in lowered
        for marker in (
            "redirecttologinpage",
            "arrow_downward",
            "back to index",
            "for any comments/enquiries/feedback",
            "for any comments enquiries feedback",
            "view all latest news on old website",
            "please view all",
            "link open intranet",
            "downloaded clicking following links",
        )
    ):
        return True
    if any(hint in lowered for hint in ROSTER_LINE_HINTS):
        return True

    line_tokens = [normalize_token(token) for token in TOKEN_PATTERN.findall(lowered)]
    if sum(COURSE_CODE_PATTERN.fullmatch(token) is not None for token in line_tokens) >= 2:
        return True
    if is_roster_like_segment([token for token in line_tokens if not is_noise_token(token)]):
        return True

    word_tokens = re.findall(r"[a-z]+", lowered)
    if not word_tokens:
        return True
    if len(word_tokens) <= 2 and all(len(token) <= 1 for token in word_tokens):
        return True
    if len(word_tokens) <= 3 and all(token in SHORT_LINE_NOISE_TOKENS or token in TITLE_NOISE_TOKENS for token in word_tokens):
        return True

    hint_count = sum(token in ANNOUNCEMENT_HINT_TOKENS for token in word_tokens)
    if len(word_tokens) <= 10 and hint_count >= 2:
        return True
    if len(word_tokens) <= 18 and hint_count >= 3:
        return True

    if lowered.startswith("download file") or lowered.startswith("clickhere"):
        return True
    if lowered.startswith("note: examples of course codes"):
        return True
    return False


def should_drop_paragraph(paragraph: str, doc_type: str) -> bool:
    lowered = paragraph.strip().lower()
    if not lowered:
        return True
    if any(phrase in lowered for phrase in PARAGRAPH_BOILERPLATE_PHRASES):
        return True

    tokens = [normalize_token(token) for token in TOKEN_PATTERN.findall(lowered)]
    tokens = [token for token in tokens if not is_noise_token(token)]
    if not tokens:
        return True

    if doc_type not in {"Academic Regulation", "Academic Program", "Course Syllabus"}:
        pedagogy_hits = sum(token in PEDAGOGY_OVERLOAD_TOKENS for token in tokens)
        domain_hits = sum(token in {"research", "project", "publication", "lab", "hostel", "student_life", "phd"} for token in tokens)
        if pedagogy_hits >= max(6, len(tokens) // 5) and domain_hits <= 3:
            return True
        return False

    metadata_hits = sum(token in SYLLABUS_METADATA_TOKENS for token in tokens)
    reference_hits = sum(token in REFERENCE_SECTION_TOKENS for token in tokens)
    promo_hits = sum(token in PROMOTIONAL_SECTION_TOKENS for token in tokens)

    if len(tokens) <= 20 and metadata_hits >= 3:
        return True
    if len(tokens) <= 30 and reference_hits >= 3:
        return True
    if metadata_hits >= max(5, len(tokens) // 4):
        return True
    if reference_hits >= max(5, len(tokens) // 4):
        return True
    if promo_hits >= max(4, len(tokens) // 5):
        return True
    return False


def clean_text(text: str) -> str:
    from dataset_generation.scraper.content_filters import clean_scraped_content

    text = clean_scraped_content(text)
    text = unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MARKDOWN_LINK_PATTERN.sub(lambda match: f" {match.group(1)} ", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = normalize_domain_phrases(text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = UUID_PATTERN.sub(" ", text)
    text = re.sub(r"/PageImages/\S+|PageImages/\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"[*`~><]+", " ", text)
    text = re.sub(r"[•●▪◆■]+", "\n", text)
    text = re.sub(r"[|]+", "\n", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)

    cleaned_lines: list[str] = []
    blank_pending = False
    for raw_line in text.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip(" -:\t")
        if not line or should_drop_line(line):
            blank_pending = True
            continue
        if blank_pending and cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")
        cleaned_lines.append(line)
        blank_pending = False

    paragraphs: list[str] = []
    for paragraph in PARAGRAPH_SPLIT_PATTERN.split("\n".join(cleaned_lines)):
        merged = re.sub(r"\s*\n\s*", " ", paragraph).strip()
        if not merged:
            continue
        merged = re.sub(r"\s*([.!?;:])\s*", r"\1 ", merged)
        merged = re.sub(r"[ \t]+", " ", merged).strip()
        if merged:
            paragraphs.append(merged)

    return "\n\n".join(paragraphs)


def is_low_signal_segment(tokens: list[str]) -> bool:
    if len(tokens) < MIN_SEGMENT_TOKENS:
        return True
    if is_roster_like_segment(tokens):
        return True
    hint_count = sum(token in ANNOUNCEMENT_HINT_TOKENS for token in tokens)
    if len(tokens) <= 20 and hint_count >= 3:
        return True
    if len(tokens) <= 36 and hint_count >= max(4, len(tokens) // 4):
        return True
    if hint_count >= max(6, len(tokens) // 3):
        return True
    if len(tokens) >= 20 and (len(set(tokens)) / len(tokens)) < 0.3:
        return True
    return False


def should_skip_token_sequence(tokens: list[str]) -> bool:
    if not tokens:
        return True
    if is_roster_like_segment(tokens):
        return True
    hint_count = sum(token in ANNOUNCEMENT_HINT_TOKENS for token in tokens)
    if len(tokens) <= 8 and hint_count >= 2:
        return True
    if len(tokens) <= 12 and hint_count >= 3:
        return True
    if len(tokens) <= 4 and len(set(tokens)) <= 2:
        return True
    return False


def tokenize_segment(text: str) -> list[str]:
    nlp = get_spacy_nlp()
    if nlp is not None:
        try:
            doc = nlp(text)
            tokens: list[str] = []
            for token in doc:
                if token.is_space or token.is_punct or token.like_num or token.like_url:
                    continue

                raw_text = token.text.strip()
                if not raw_text:
                    continue

                if token.ent_type_ == "PERSON" and raw_text[:1].isupper():
                    continue

                lemma = raw_text
                if token.lemma_ and token.lemma_ != "-PRON-":
                    lemma = token.lemma_

                for mapped in expand_mapped_token(normalize_token_basic(lemma)):
                    if is_noise_token(mapped):
                        continue
                    tokens.append(mapped)
            return tokens
        except Exception:
            pass

    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text):
        for mapped in expand_normalized_token(token):
            if is_noise_token(mapped):
                continue
            tokens.append(mapped)
    return tokens


def should_use_title_tokens(document: CorpusDocument, title_tokens: list[str]) -> bool:
    if len(title_tokens) < 3 or len(title_tokens) > 12:
        return False
    title = str(document.metadata.get("title", "")).lower()
    doc_type = str(document.metadata.get("doc_type", ""))
    if any(hint in title for hint in EXCLUDED_TITLE_HINTS):
        return False
    if any(hint in title for hint in ("faculty members", "research highlights", "coordination committee")):
        return False
    anchor_count = sum(token in TITLE_ANCHOR_TOKENS for token in title_tokens)
    if doc_type in {"Academic Regulation", "Academic Program", "Course Syllabus", "Department"}:
        return anchor_count >= 1
    return anchor_count >= 2


def should_consider_phrase_pair(left: str, right: str) -> bool:
    if left in PHRASE_BLOCKLIST_TOKENS or right in PHRASE_BLOCKLIST_TOKENS:
        return False
    if left in ENGLISH_STOP_WORDS or right in ENGLISH_STOP_WORDS:
        return False
    if left in NOISE_TOKENS or right in NOISE_TOKENS:
        return False
    if "_" in left or "_" in right:
        return False
    if len(left) < 3 or len(right) < 3:
        return False
    if any(char.isdigit() for char in left + right):
        return False
    if left not in PHRASE_SEED_TOKENS and right not in PHRASE_SEED_TOKENS:
        return False
    return True


def learn_bigram_phrases(token_sequences: list[list[str]]) -> tuple[dict[tuple[str, str], str], list[dict[str, float | int | str]]]:
    unigram_counts: Counter[str] = Counter()
    bigram_counts: Counter[tuple[str, str]] = Counter()
    total_tokens = 0

    for tokens in token_sequences:
        if not tokens:
            continue
        unigram_counts.update(tokens)
        total_tokens += len(tokens)
        for left, right in zip(tokens, tokens[1:]):
            if should_consider_phrase_pair(left, right):
                bigram_counts[(left, right)] += 1

    scored_phrases: list[dict[str, float | int | str]] = []
    for (left, right), pair_count in bigram_counts.items():
        if pair_count < PHRASE_MIN_COUNT:
            continue
        if unigram_counts[left] < PHRASE_MIN_UNIGRAM_COUNT or unigram_counts[right] < PHRASE_MIN_UNIGRAM_COUNT:
            continue
        if left not in PHRASE_SEED_TOKENS and unigram_counts[left] < 20:
            continue
        if right not in PHRASE_SEED_TOKENS and unigram_counts[right] < 20:
            continue

        score = ((pair_count - PHRASE_MIN_COUNT + 1) * total_tokens) / max(unigram_counts[left] * unigram_counts[right], 1)
        if score < PHRASE_MIN_SCORE:
            continue

        scored_phrases.append(
            {
                "left": left,
                "right": right,
                "phrase": f"{left}_{right}",
                "count": int(pair_count),
                "score": round(float(score), 4),
            }
        )

    scored_phrases.sort(
        key=lambda item: (
            -float(item["score"]),
            -int(item["count"]),
            str(item["phrase"]),
        )
    )
    scored_phrases = scored_phrases[:PHRASE_MAX_VOCAB]
    phrase_map = {
        (str(item["left"]), str(item["right"])): str(item["phrase"])
        for item in scored_phrases
    }
    return phrase_map, scored_phrases


def apply_bigram_phrases(tokens: list[str], phrase_map: dict[tuple[str, str], str]) -> tuple[list[str], int]:
    if not phrase_map or len(tokens) < 2:
        return tokens, 0

    merged_tokens: list[str] = []
    merge_events = 0
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens):
            phrase = phrase_map.get((tokens[index], tokens[index + 1]))
            if phrase:
                merged_tokens.append(phrase)
                merge_events += 1
                index += 2
                continue
        merged_tokens.append(tokens[index])
        index += 1

    return merged_tokens, merge_events


def build_token_sequences(text: str, *, doc_type: str) -> list[list[str]]:
    sequences: list[list[str]] = []
    for paragraph in PARAGRAPH_SPLIT_PATTERN.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if should_drop_paragraph(paragraph, doc_type):
            continue
        units = [unit.strip() for unit in SENTENCE_SPLIT_PATTERN.split(paragraph) if unit.strip()]
        if len(units) == 1:
            units = [paragraph]
        for unit in units:
            tokens = tokenize_segment(unit)
            if should_skip_token_sequence(tokens):
                continue
            sequences.append(tokens)
    return sequences


def split_long_sequence(tokens: list[str]) -> list[list[str]]:
    if len(tokens) <= MAX_SEGMENT_TOKENS:
        return [tokens]

    stride = max(1, MAX_SEGMENT_TOKENS - SEGMENT_OVERLAP_TOKENS)
    chunks: list[list[str]] = []
    for start in range(0, len(tokens), stride):
        chunk = tokens[start : start + MAX_SEGMENT_TOKENS]
        if len(chunk) < MIN_SEGMENT_TOKENS:
            break
        chunks.append(chunk)
        if start + MAX_SEGMENT_TOKENS >= len(tokens):
            break
    return chunks


def pack_token_sequences(sequences: list[list[str]]) -> list[list[str]]:
    packed_segments: list[list[str]] = []
    buffer: list[str] = []

    for sequence in sequences:
        if not sequence:
            continue

        if len(sequence) >= TARGET_SEGMENT_TOKENS:
            if len(buffer) >= MIN_SEGMENT_TOKENS and not is_low_signal_segment(buffer):
                packed_segments.append(buffer.copy())
            buffer = []
            for chunk in split_long_sequence(sequence):
                if len(chunk) >= MIN_SEGMENT_TOKENS and not is_low_signal_segment(chunk):
                    packed_segments.append(chunk)
            continue

        if buffer and len(buffer) + len(sequence) > MAX_SEGMENT_TOKENS:
            if len(buffer) >= MIN_SEGMENT_TOKENS and not is_low_signal_segment(buffer):
                packed_segments.append(buffer.copy())
                buffer = buffer[-SEGMENT_OVERLAP_TOKENS:].copy()
            else:
                buffer = []

        buffer.extend(sequence)
        if len(buffer) >= TARGET_SEGMENT_TOKENS:
            if not is_low_signal_segment(buffer):
                packed_segments.append(buffer.copy())
            buffer = buffer[-SEGMENT_OVERLAP_TOKENS:].copy()

    if len(buffer) >= MIN_SEGMENT_TOKENS and not is_low_signal_segment(buffer):
        packed_segments.append(buffer)

    return packed_segments


def limit_segments_per_document(document: CorpusDocument, segments: list[list[str]]) -> list[list[str]]:
    max_segments = DOC_TYPE_SEGMENT_LIMITS.get(str(document.metadata.get("doc_type", "")), MAX_SEGMENTS_PER_DOCUMENT)
    if len(segments) <= max_segments:
        return segments

    step = len(segments) / max_segments
    sampled = [segments[min(int(index * step), len(segments) - 1)] for index in range(max_segments)]
    deduplicated: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for segment in sampled:
        key = tuple(segment)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(segment)
    return deduplicated


def build_segments_for_document(document: CorpusDocument) -> list[list[str]]:
    normalized = clean_text(document.content)
    if not normalized:
        return []

    token_sequences = build_token_sequences(normalized, doc_type=str(document.metadata.get("doc_type", "")))
    title_tokens = title_to_tokens(str(document.metadata.get("title", "")))
    if should_use_title_tokens(document, title_tokens):
        token_sequences.insert(0, title_tokens)

    processed_segments: list[list[str]] = []
    seen_segments: set[tuple[str, ...]] = set()
    for segment in pack_token_sequences(token_sequences):
        if len(segment) < MIN_SEGMENT_TOKENS or is_low_signal_segment(segment):
            continue
        key = tuple(segment)
        if key in seen_segments:
            continue
        seen_segments.add(key)
        processed_segments.append(segment)

    return limit_segments_per_document(document, processed_segments)


def collect_candidate_segments(
    documents: list[CorpusDocument],
    *,
    use_langid: bool,
) -> tuple[list[CandidateSegment], int, Counter, Counter, Counter]:
    candidate_segments: list[CandidateSegment] = []
    kept_doc_types: Counter = Counter()
    kept_departments: Counter = Counter()
    skipped_reasons: Counter = Counter()
    retained_documents = 0

    for document in documents:
        keep, reason = should_keep_document(document, use_langid=use_langid)
        if not keep:
            skipped_reasons[reason] += 1
            continue

        segments = build_segments_for_document(document)
        if not segments:
            skipped_reasons["no_valid_segments"] += 1
            continue

        retained_documents += 1
        metadata = document.metadata
        kept_doc_types[str(metadata.get("doc_type", "UNKNOWN"))] += 1
        kept_departments[str(metadata.get("department", "Institute"))] += 1

        for segment_index, tokens in enumerate(segments):
            candidate_segments.append(
                CandidateSegment(
                    metadata=metadata,
                    segment_index=segment_index,
                    tokens=tokens,
                )
            )

    return candidate_segments, retained_documents, kept_doc_types, kept_departments, skipped_reasons


def identify_training_exclusion_tokens(
    token_sequences: list[list[str]],
) -> tuple[set[str], list[dict[str, float | int | str]]]:
    token_counts: Counter[str] = Counter()
    token_document_counts: Counter[str] = Counter()
    total_tokens = 0

    for tokens in token_sequences:
        if not tokens:
            continue
        token_counts.update(tokens)
        token_document_counts.update(set(tokens))
        total_tokens += len(tokens)

    total_sequences = max(len(token_sequences), 1)
    excluded_tokens: set[str] = set()
    diagnostics: list[dict[str, float | int | str]] = []

    for token, count in token_counts.items():
        if token in PROTECTED_TRAINING_TOKENS:
            continue

        frequency_ratio = count / max(total_tokens, 1)
        document_ratio = token_document_counts[token] / total_sequences

        should_drop = token in FORCED_DROP_TRAINING_TOKENS
        if not should_drop and frequency_ratio >= TRAINING_DROP_FREQ_RATIO and document_ratio >= TRAINING_DROP_DOC_RATIO:
            should_drop = True

        if not should_drop:
            continue

        excluded_tokens.add(token)
        diagnostics.append(
            {
                "token": token,
                "count": int(count),
                "frequency_ratio": round(float(frequency_ratio), 5),
                "document_ratio": round(float(document_ratio), 5),
            }
        )

    diagnostics.sort(
        key=lambda item: (
            -int(item["count"]),
            str(item["token"]),
        )
    )
    return excluded_tokens, diagnostics


def write_dataset_statistics(stats_path: Path, stats: dict) -> None:
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def describe_tokenizer_backend() -> str:
    nlp = get_spacy_nlp()
    if nlp is None:
        return "regex_only"
    if getattr(nlp, "meta", None):
        model_name = nlp.meta.get("name") or SPACY_MODEL_NAME
    else:
        model_name = SPACY_MODEL_NAME
    return f"spacy:{model_name}"


def write_mapping_audit(
    mapping_path: Path,
    learned_phrases: list[dict[str, float | int | str]] | None = None,
    dropped_training_tokens: list[dict[str, float | int | str]] | None = None,
) -> None:
    payload = {
        "phrase_replacements": [{"pattern": pattern, "replacement": replacement} for pattern, replacement in PHRASE_REPLACEMENTS],
        "token_normalization": TOKEN_NORMALIZATION,
        "noise_tokens": sorted(NOISE_TOKENS),
        "announcement_hint_tokens": sorted(ANNOUNCEMENT_HINT_TOKENS),
        "excluded_title_hints": list(EXCLUDED_TITLE_HINTS),
        "excluded_url_hints": list(EXCLUDED_URL_HINTS),
        "forced_include_hints": list(FORCE_INCLUDE_HINTS),
        "doc_type_keep": sorted(DOC_TYPE_KEEP),
        "segment_policy": {
            "min_segment_tokens": MIN_SEGMENT_TOKENS,
            "target_segment_tokens": TARGET_SEGMENT_TOKENS,
            "max_segment_tokens": MAX_SEGMENT_TOKENS,
            "segment_overlap_tokens": SEGMENT_OVERLAP_TOKENS,
            "max_segments_per_document": MAX_SEGMENTS_PER_DOCUMENT,
        },
        "data_driven_phrase_policy": {
            "phrase_min_count": PHRASE_MIN_COUNT,
            "phrase_min_unigram_count": PHRASE_MIN_UNIGRAM_COUNT,
            "phrase_min_score": PHRASE_MIN_SCORE,
            "phrase_max_vocab": PHRASE_MAX_VOCAB,
            "phrase_seed_tokens": sorted(PHRASE_SEED_TOKENS),
            "phrase_blocklist_tokens": sorted(PHRASE_BLOCKLIST_TOKENS),
        },
        "training_token_pruning_policy": {
            "forced_drop_tokens": sorted(FORCED_DROP_TRAINING_TOKENS),
            "protected_training_tokens": sorted(PROTECTED_TRAINING_TOKENS),
            "training_drop_freq_ratio": TRAINING_DROP_FREQ_RATIO,
            "training_drop_doc_ratio": TRAINING_DROP_DOC_RATIO,
        },
        "learned_phrases_top_100": (learned_phrases or [])[:100],
        "dropped_training_tokens_top_100": (dropped_training_tokens or [])[:100],
    }
    mapping_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_frequency_visualization(wordcloud_img: Path, token_frequencies: Counter) -> None:
    stop_words = set(STOPWORDS) | set(ENGLISH_STOP_WORDS)
    filtered_freq = {word: count for word, count in token_frequencies.items() if word not in stop_words}
    if not filtered_freq:
        filtered_freq = dict(token_frequencies.most_common(200))

    plt.figure(figsize=(12, 6))
    if HAS_WORDCLOUD:
        word_cloud = WordCloud(width=1000, height=500, background_color="white", max_words=200)
        word_cloud.generate_from_frequencies(filtered_freq)
        plt.imshow(word_cloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("IIT Jodhpur Corpus Word Cloud")
    else:
        top_items = sorted(filtered_freq.items(), key=lambda item: item[1], reverse=True)[:20]
        words = [item[0] for item in reversed(top_items)]
        counts = [item[1] for item in reversed(top_items)]
        plt.barh(words, counts, color="steelblue")
        plt.title("IIT Jodhpur Top Tokens (WordCloud package unavailable)")
        plt.xlabel("Frequency")
        plt.tight_layout()

    plt.savefig(wordcloud_img, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_file)
    fallback_input_file = Path(args.input_file)
    processed_corpus_path = Path(args.processed_corpus)
    segments_jsonl_path = Path(args.segments_jsonl)
    stats_path = Path(args.stats_file)
    mapping_path = Path(args.mapping_file)
    wordcloud_img = Path(args.wordcloud_img)

    documents, corpus_source = collect_corpus_documents(output_dir, manifest_path, fallback_input_file)
    if not documents:
        print(
            "Error: no corpus documents found. Expected manifest/doc JSON files under "
            f"{output_dir} or fallback text at {fallback_input_file}."
        )
        return

    print(f"Loaded {len(documents)} raw documents from {corpus_source}.")

    processed_corpus_path.parent.mkdir(parents=True, exist_ok=True)
    segments_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    wordcloud_img.parent.mkdir(parents=True, exist_ok=True)

    candidate_segments, retained_documents, kept_doc_types, kept_departments, skipped_reasons = collect_candidate_segments(
        documents,
        use_langid=HAS_LANGID and not args.disable_langid,
    )
    # Learn common academic bigrams first, then merge them back into the segments.
    phrase_map, learned_phrases = learn_bigram_phrases([segment.tokens for segment in candidate_segments])
    print(f"Learned {len(learned_phrases)} academic bigrams for corpus enrichment.")

    phrased_segments: list[tuple[CandidateSegment, list[str], int]] = []
    for candidate in candidate_segments:
        merged_tokens, merge_count = apply_bigram_phrases(candidate.tokens, phrase_map)
        normalized_tokens: list[str] = []
        for token in merged_tokens:
            normalized_tokens.extend(expand_normalized_token(token))
        phrased_segments.append((candidate, normalized_tokens, merge_count))

    training_exclusion_tokens, training_exclusion_diagnostics = identify_training_exclusion_tokens(
        [tokens for _candidate, tokens, _merge_count in phrased_segments]
    )
    # Very common boilerplate still hurts embedding quality, so drop it before writing the corpus.
    print(f"Dropping {len(training_exclusion_tokens)} high-frequency boilerplate tokens from training sequences.")

    token_frequencies: Counter = Counter()
    document_lengths: list[int] = []
    global_seen_segment_hashes: set[str] = set()
    retained_segments = 0
    duplicate_segments_skipped = 0
    phrase_segments = 0
    phrase_merge_events = 0

    with processed_corpus_path.open("w", encoding="utf-8") as corpus_handle, segments_jsonl_path.open(
        "w", encoding="utf-8"
    ) as segments_handle:
        for candidate, merged_tokens, merge_count in phrased_segments:
            tokens = [token for token in merged_tokens if token not in training_exclusion_tokens]
            if len(tokens) < MIN_SEGMENT_TOKENS or is_low_signal_segment(tokens):
                continue

            metadata = candidate.metadata
            segment_text = " ".join(tokens)
            segment_hash = sha1(segment_text.encode("utf-8")).hexdigest()
            # This second dedup step avoids writing near-identical training rows.
            if segment_hash in global_seen_segment_hashes:
                duplicate_segments_skipped += 1
                continue
            global_seen_segment_hashes.add(segment_hash)

            if merge_count > 0:
                phrase_segments += 1
                phrase_merge_events += merge_count

            retained_segments += 1
            token_frequencies.update(tokens)
            document_lengths.append(len(tokens))
            corpus_handle.write(segment_text + "\n")
            segments_handle.write(
                json.dumps(
                    {
                        "segment_id": f"{metadata.get('id', 'doc')}-{candidate.segment_index}",
                        "source_id": metadata.get("id"),
                        "source_url": metadata.get("source_url"),
                        "title": metadata.get("title"),
                        "doc_type": metadata.get("doc_type"),
                        "department": metadata.get("department"),
                        "token_count": len(tokens),
                        "phrase_merges": merge_count,
                        "text": segment_text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total_tokens = sum(token_frequencies.values())
    vocabulary_size = len(token_frequencies)
    top_10_tokens = token_frequencies.most_common(10)
    tokenizer_backend = describe_tokenizer_backend()

    print("=" * 52)
    print("DATASET STATISTICS")
    print(f"Tokenizer Backend   : {tokenizer_backend}")
    print(f"Raw Documents       : {len(documents)}")
    print(f"Retained Documents  : {retained_documents}")
    print(f"Training Segments   : {retained_segments}")
    print(f"Skipped Duplicates  : {duplicate_segments_skipped}")
    print(f"Phrase Segments     : {phrase_segments}")
    print(f"Phrase Merge Events : {phrase_merge_events}")
    print(f"Dropped Tokens      : {len(training_exclusion_tokens)}")
    print(f"Total Tokens        : {total_tokens}")
    print(f"Vocabulary Size     : {vocabulary_size}")
    print("Top 10 Tokens       : " + ", ".join(f"{token} ({count})" for token, count in top_10_tokens))
    print("=" * 52)

    stats_payload = {
        "corpus_source": corpus_source,
        "tokenizer_backend": tokenizer_backend,
        "raw_documents": len(documents),
        "total_documents": retained_documents,
        "training_segments": retained_segments,
        "duplicate_segments_skipped": duplicate_segments_skipped,
        "learned_phrases_count": len(learned_phrases),
        "phrase_segments": phrase_segments,
        "phrase_merge_events": phrase_merge_events,
        "training_exclusion_tokens_count": len(training_exclusion_tokens),
        "total_tokens": total_tokens,
        "vocabulary_size": vocabulary_size,
        "avg_document_length_tokens": round(mean(document_lengths), 2) if document_lengths else 0.0,
        "median_document_length_tokens": float(median(document_lengths)) if document_lengths else 0.0,
        "top_10_tokens": top_10_tokens,
        "top_25_tokens": token_frequencies.most_common(25),
        "learned_phrases_top_25": learned_phrases[:25],
        "dropped_training_tokens_top_50": training_exclusion_diagnostics[:50],
        "kept_doc_types": dict(kept_doc_types),
        "kept_departments_top_20": dict(kept_departments.most_common(20)),
        "skipped_reasons": dict(skipped_reasons),
    }
    write_dataset_statistics(stats_path, stats_payload)
    # Keep a small audit trail so it is easy to inspect what normalization changed.
    write_mapping_audit(
        mapping_path,
        learned_phrases=learned_phrases,
        dropped_training_tokens=training_exclusion_diagnostics,
    )

    print("Generating word frequency visualization...")
    generate_frequency_visualization(wordcloud_img, token_frequencies)

    print(f"Processed corpus saved to {processed_corpus_path}")
    print(f"Per-segment audit saved to {segments_jsonl_path}")
    print(f"Dataset statistics saved to {stats_path}")
    print(f"Normalization map saved to {mapping_path}")
    if HAS_WORDCLOUD:
        print(f"Word cloud saved to {wordcloud_img}")
    else:
        print(f"WordCloud package not installed; saved top-token fallback plot to {wordcloud_img}")


if __name__ == "__main__":
    main()
