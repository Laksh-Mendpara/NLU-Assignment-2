from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha1
from html import unescape
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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


TOKEN_PATTERN = re.compile(r"[a-z]+(?:[a-z0-9_'-]*[a-z0-9]+)?")
SEGMENT_SPLIT_PATTERN = re.compile(r"(?:\n+|(?<=[.!?;:])\s+)")
URL_PATTERN = re.compile(r"http\S+|www\.\S+")
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
UUID_PATTERN = re.compile(r"\b[a-f0-9]{8,}(?:-[a-f0-9]{4,})+\b", flags=re.IGNORECASE)

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
    (r"\bdigital humanities\b", "digital_humanities"),
    (r"\bcyber[\s-]*physical systems\b", "cyber_physical_systems"),
    (r"\binternet of things\b", "internet_of_things"),
    (r"\bmachine learning\b", "machine_learning"),
    (r"\bdeep learning\b", "deep_learning"),
    (r"\bacademic regulations?\b", "academic_regulation"),
    (r"\boffice of academic affairs\b", "office_of_academic_affairs"),
    (r"\boffice of academics affairs\b", "office_of_academic_affairs"),
    (r"\bschool of management(?:\s+and\s+|[\s&]+)?entrepreneurship\b", "school_of_management_entrepreneurship"),
    (r"\bph[\s./-]*d\b", "phd"),
    (r"\bb[\s./-]*tech\b", "btech"),
    (r"\bm[\s./-]*tech\b", "mtech"),
    (r"\bb[\s./-]*sc\b", "bsc"),
    (r"\bm[\s./-]*sc\b", "msc"),
    (r"\bp[\s./-]*g\b", "pg"),
    (r"\bu[\s./-]*g\b", "ug"),
)

TOKEN_NORMALIZATION = {
    "programmes": "program",
    "programme": "program",
    "programs": "program",
    "courses": "course",
    "students": "student",
    "faculties": "faculty",
    "departments": "department",
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
    "high-quality": "high_quality",
    "cutting-edge": "cutting_edge",
    "state-of-the-art": "state_of_the_art",
    "mtech-phd": "mtech_phd",
}

NOISE_TOKENS = {
    "pageimages",
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
    "page",
    "image",
    "images",
    "ay",
    "iitj",
}

NOISE_PHRASES = (
    "for any comments enquiries feedback please email the wim",
    "last updated",
    "download file",
    "javascript void",
    "pageimages gallery",
    "dsl dslid",
    "click to know more",
)

EXCLUDED_TITLE_HINTS = (
    "contact",
    "people",
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
)

EXCLUDED_URL_HINTS = (
    "/contact",
    "/people",
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
)

FORCE_INCLUDE_HINTS = (
    "research highlight",
    "research project",
    "curriculum",
    "course booklet",
    "program structure",
    "program pathway",
    "academic regulation",
    "faculty position",
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
}

DOC_TYPE_MIN_CONTENT = {
    "Research": 200,
    "Academic Regulation": 300,
    "Academic Program": 250,
    "Course Syllabus": 250,
    "Faculty Profile": 150,
    "Department": 250,
}

ANNOUNCEMENT_HINT_TOKENS = {
    "admission",
    "shortlisted",
    "selected",
    "candidate",
    "reporting",
    "merit",
    "semester",
    "round",
    "interview",
    "result",
    "exam",
}

TITLE_NOISE_TOKENS = {"iit", "iitj", "jodhpur", "indian", "institute", "technology"}
MIN_SEGMENT_TOKENS = 5
MAX_SEGMENT_TOKENS = 80
MIN_SEGMENT_CHARS = 35


@dataclass
class CorpusDocument:
    metadata: dict
    content: str
    source_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean IITJ corpus for Word2Vec training.")
    parser.add_argument("--output-dir", default="output", help="Directory containing scraped artifacts.")
    parser.add_argument("--manifest-file", default="output/manifest.json", help="Manifest file from the scraper.")
    parser.add_argument(
        "--input-file",
        default="output/data.txt",
        help="Fallback aggregated raw text corpus used only if manifest/doc JSON files are unavailable.",
    )
    parser.add_argument("--processed-corpus", default="output/processed_corpus.txt", help="Tokenized corpus output path.")
    parser.add_argument("--segments-jsonl", default="output/processed_segments.jsonl", help="Per-segment metadata output.")
    parser.add_argument("--stats-file", default="output/dataset_stats.json", help="Dataset statistics JSON output path.")
    parser.add_argument("--mapping-file", default="output/normalization_map.json", help="Normalization/mapping audit file.")
    parser.add_argument("--wordcloud-img", default="output/wordcloud.png", help="Visualization output path.")
    parser.add_argument("--disable-langid", action="store_true", help="Skip optional language filtering.")
    return parser.parse_args()


def normalize_domain_phrases(text: str) -> str:
    for pattern, replacement in PHRASE_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def normalize_token(token: str) -> str:
    return TOKEN_NORMALIZATION.get(token, token)


def is_noise_token(token: str) -> bool:
    if token in NOISE_TOKENS:
        return True
    if len(token) <= 1:
        return True
    if token.isdigit():
        return True
    if token.startswith("dsl") or token.startswith("pageimage"):
        return True
    if any(char.isdigit() for char in token) and len(token) >= 6:
        return True
    if token.count("-") >= 2:
        return True
    if re.fullmatch(r"[a-f0-9_/-]{8,}", token):
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
        content = raw_doc.get("content", "")
        documents.append(CorpusDocument(metadata=merged_metadata, content=content, source_path=source_path))
    return documents


def load_documents_from_directories(output_dir: Path) -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    for subdir in ("data", "pdfs"):
        directory = output_dir / subdir
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            raw_doc = json.loads(path.read_text(encoding="utf-8"))
            documents.append(
                CorpusDocument(
                    metadata=raw_doc.get("metadata", {}),
                    content=raw_doc.get("content", ""),
                    source_path=path,
                )
            )
    return documents


def collect_corpus_documents(output_dir: Path, manifest_path: Path, fallback_input_file: Path) -> tuple[list[CorpusDocument], str]:
    if manifest_path.exists():
        documents = load_documents_from_manifest(manifest_path, output_dir)
        if documents:
            return documents, "manifest"

    documents = load_documents_from_directories(output_dir)
    if documents:
        return documents, "json_directories"

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


def should_keep_document(document: CorpusDocument, use_langid: bool) -> tuple[bool, str]:
    metadata = document.metadata
    title = str(metadata.get("title", "")).lower()
    source_url = str(metadata.get("source_url", "")).lower()
    doc_type = str(metadata.get("doc_type", ""))
    content = document.content or ""
    content_length = len(content.strip())

    if use_langid and not is_english_text(" ".join([title, content[:1500]])):
        return False, "non_english"

    if doc_type not in DOC_TYPE_KEEP and not any(hint in title for hint in FORCE_INCLUDE_HINTS):
        return False, f"excluded_doc_type:{doc_type or 'unknown'}"

    if any(hint in title for hint in EXCLUDED_TITLE_HINTS) or any(hint in source_url for hint in EXCLUDED_URL_HINTS):
        if not any(hint in title for hint in FORCE_INCLUDE_HINTS):
            return False, "excluded_by_title_or_url"

    minimum_length = DOC_TYPE_MIN_CONTENT.get(doc_type, 250)
    if content_length < minimum_length:
        return False, f"too_short:{doc_type or 'unknown'}"

    return True, "kept"


def title_to_tokens(title: str) -> list[str]:
    normalized = clean_text(title).lower()
    tokens = []
    for token in TOKEN_PATTERN.findall(normalized):
        mapped = normalize_token(token)
        if mapped in TITLE_NOISE_TOKENS or is_noise_token(mapped):
            continue
        tokens.append(mapped)
    return tokens


def clean_text(text: str) -> str:
    text = unescape(text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = normalize_domain_phrases(text)
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = UUID_PATTERN.sub(" ", text)
    text = re.sub(r"/PageImages/\S+|PageImages/\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"[*`~><]+", " ", text)
    text = re.sub(r"[|]+", "\n", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    for phrase in NOISE_PHRASES:
        text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s*([.!?;:])\s*", r"\1 ", text)
    text = re.sub(r"\n\s*", "\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def is_low_signal_segment(tokens: list[str]) -> bool:
    hint_count = sum(token in ANNOUNCEMENT_HINT_TOKENS for token in tokens)
    if len(tokens) <= 12 and hint_count >= 3:
        return True
    if len(tokens) <= 20 and hint_count >= 4:
        return True
    return False


def tokenize_segment(text: str) -> list[str]:
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(text):
        mapped = normalize_token(token)
        if is_noise_token(mapped):
            continue
        tokens.append(mapped)
    return tokens


def build_segments_for_document(document: CorpusDocument) -> list[list[str]]:
    normalized = clean_text(document.content).lower()
    raw_segments = SEGMENT_SPLIT_PATTERN.split(normalized)
    processed_segments: list[list[str]] = []
    seen_segments: set[tuple[str, ...]] = set()

    title_tokens = title_to_tokens(str(document.metadata.get("title", "")))
    if len(title_tokens) >= MIN_SEGMENT_TOKENS:
        processed_segments.append(title_tokens)
        seen_segments.add(tuple(title_tokens))

    for segment in raw_segments:
        if len(segment.strip()) < MIN_SEGMENT_CHARS:
            continue
        tokens = tokenize_segment(segment)
        if len(tokens) < MIN_SEGMENT_TOKENS or is_low_signal_segment(tokens):
            continue

        for start in range(0, len(tokens), MAX_SEGMENT_TOKENS):
            chunk = tokens[start : start + MAX_SEGMENT_TOKENS]
            if len(chunk) < MIN_SEGMENT_TOKENS or is_low_signal_segment(chunk):
                continue
            chunk_key = tuple(chunk)
            if chunk_key in seen_segments:
                continue
            seen_segments.add(chunk_key)
            processed_segments.append(chunk)

    return processed_segments


def write_dataset_statistics(stats_path: Path, stats: dict) -> None:
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def write_mapping_audit(mapping_path: Path) -> None:
    payload = {
        "phrase_replacements": [{"pattern": pattern, "replacement": replacement} for pattern, replacement in PHRASE_REPLACEMENTS],
        "token_normalization": TOKEN_NORMALIZATION,
        "noise_tokens": sorted(NOISE_TOKENS),
        "excluded_title_hints": list(EXCLUDED_TITLE_HINTS),
        "excluded_url_hints": list(EXCLUDED_URL_HINTS),
        "forced_include_hints": list(FORCE_INCLUDE_HINTS),
        "doc_type_keep": sorted(DOC_TYPE_KEEP),
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

    token_frequencies: Counter = Counter()
    document_lengths: list[int] = []
    kept_doc_types: Counter = Counter()
    skipped_reasons: Counter = Counter()
    kept_departments: Counter = Counter()
    global_seen_segment_hashes: set[str] = set()

    retained_documents = 0
    retained_segments = 0
    duplicate_segments_skipped = 0

    with processed_corpus_path.open("w", encoding="utf-8") as corpus_handle, segments_jsonl_path.open(
        "w", encoding="utf-8"
    ) as segments_handle:
        for document in documents:
            keep, reason = should_keep_document(document, use_langid=HAS_LANGID and not args.disable_langid)
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
                segment_text = " ".join(tokens)
                segment_hash = sha1(segment_text.encode("utf-8")).hexdigest()
                if segment_hash in global_seen_segment_hashes:
                    duplicate_segments_skipped += 1
                    continue
                global_seen_segment_hashes.add(segment_hash)

                retained_segments += 1
                token_frequencies.update(tokens)
                document_lengths.append(len(tokens))
                corpus_handle.write(segment_text + "\n")
                segments_handle.write(
                    json.dumps(
                        {
                            "segment_id": f"{metadata.get('id', 'doc')}-{segment_index}",
                            "source_id": metadata.get("id"),
                            "source_url": metadata.get("source_url"),
                            "title": metadata.get("title"),
                            "doc_type": metadata.get("doc_type"),
                            "department": metadata.get("department"),
                            "token_count": len(tokens),
                            "text": segment_text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    total_tokens = sum(token_frequencies.values())
    vocabulary_size = len(token_frequencies)

    print("=" * 52)
    print("DATASET STATISTICS")
    print(f"Raw Documents       : {len(documents)}")
    print(f"Retained Documents  : {retained_documents}")
    print(f"Training Segments   : {retained_segments}")
    print(f"Skipped Duplicates  : {duplicate_segments_skipped}")
    print(f"Total Tokens        : {total_tokens}")
    print(f"Vocabulary Size     : {vocabulary_size}")
    print("=" * 52)

    stats_payload = {
        "corpus_source": corpus_source,
        "raw_documents": len(documents),
        "total_documents": retained_documents,
        "training_segments": retained_segments,
        "duplicate_segments_skipped": duplicate_segments_skipped,
        "total_tokens": total_tokens,
        "vocabulary_size": vocabulary_size,
        "avg_document_length_tokens": round(mean(document_lengths), 2) if document_lengths else 0.0,
        "median_document_length_tokens": float(median(document_lengths)) if document_lengths else 0.0,
        "top_25_tokens": token_frequencies.most_common(25),
        "kept_doc_types": dict(kept_doc_types),
        "kept_departments_top_20": dict(kept_departments.most_common(20)),
        "skipped_reasons": dict(skipped_reasons),
    }
    write_dataset_statistics(stats_path, stats_payload)
    write_mapping_audit(mapping_path)

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
