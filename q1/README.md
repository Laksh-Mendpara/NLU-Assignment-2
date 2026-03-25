# IITJ Web Scraper

A comprehensive web scraper for collecting text data from the Indian Institute of Technology Jodhpur (iitj.ac.in) domain ecosystem. Designed for building LLM/RAG training corpora.

## Features

- **Full domain crawling** — BFS traversal across `*.iitj.ac.in` (main site, departments, schools, centers, admin offices)
- **PDF extraction** — Automatically detects & downloads PDFs, converts to Markdown via `pymupdf4llm`
- **Smart content extraction** — `trafilatura` + `BeautifulSoup` fallback strips navigation/footer boilerplate
- **Deduplication** — MD5 hash-based content dedup prevents storing identical pages
- **Metadata & provenance** — Every document tagged with source URL, doc type, department, timestamp
- **Resume capability** — Saves crawl state periodically; resume with `--resume`
- **Polite crawling** — Configurable delay, concurrency limits, custom User-Agent

## Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Dependencies are already installed via uv
# If needed: uv add aiohttp beautifulsoup4 trafilatura pymupdf4llm lxml aiofiles
```

## Usage

```bash
# Quick test run (5 pages)
python main.py --max-pages 5 --delay 2

# Full domain crawl
python main.py

# Resume an interrupted crawl
python main.py --resume

# Custom settings
python main.py --output-dir ./my_data --concurrency 10 --delay 0.5

# Verbose debug output
python main.py --max-pages 10 -v
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir, -o` | `./output` | Output directory |
| `--max-pages, -m` | `0` (unlimited) | Max pages to crawl |
| `--delay, -d` | `1.0` | Seconds between requests |
| `--concurrency, -c` | `5` | Max simultaneous connections |
| `--resume, -r` | — | Resume from saved state |
| `--verbose, -v` | — | Debug-level logging |
| `--skip-sitemap` | — | Skip XML sitemap parsing |

## Output Structure

```
output/
├── data/                  # One JSON file per scraped page/PDF
│   ├── a1b2c3d4e5f6.json
│   ├── f7e8d9c0b1a2.json
│   └── ...
├── manifest.json          # Index of all documents with metadata
└── crawler_state.json     # Saved state for resume
```

### Document Format

Each JSON file contains:

```json
{
  "metadata": {
    "id": "a1b2c3d4e5f67890",
    "source_url": "https://iitj.ac.in/office-of-academics/en/Academic-Regulations",
    "title": "Academic Regulations | IIT Jodhpur",
    "doc_type": "Academic Regulation",
    "department": "Office of Academic Affairs",
    "timestamp": "2026-03-23T12:00:00+00:00",
    "content_hash": "d41d8cd98f00b204e9800998ecf8427e",
    "content_length": 15234
  },
  "content": "# Academic Regulations\n\nThe academic regulations..."
}
```

## Architecture

```
main.py                    # CLI entrypoint & orchestrator
config/
  settings.py              # Seeds, domain rules, rate limits
scraper/
  sitemap_parser.py        # XML sitemap + A-Z index discovery
  crawler.py               # Async BFS crawler with retry logic
  html_extractor.py        # HTML → clean Markdown (trafilatura + BS4)
  pdf_extractor.py         # PDF → Markdown (pymupdf4llm)
  metadata.py              # Provenance metadata builder
  dedup.py                 # MD5 hash-based deduplication
```

## Data Sources

The scraper covers all public `iitj.ac.in` subdomains:

| Category | Examples |
|----------|----------|
| Main portal | `iitj.ac.in` — Institute info, news, announcements |
| Departments | CSE, EE, ME, Chemistry, Physics, Mathematics, etc. |
| Schools | AI & Data Science, Design, Liberal Arts, Management |
| Centers | CET, CETSD, CRF, AIOT Fab Facility |
| Admin offices | Director, Registrar, Academics, Students |
| Repositories | `ir.iitj.ac.in` — theses, annual reports |
| Library | `library.iitj.ac.in` |
| Placements | `spc.iitj.ac.in` |
| PDFs | Academic regulations, Senate minutes, annual reports |
