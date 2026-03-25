# CSL7640 Natural Language Understanding Assignment 2

This repository contains both questions of Assignment 2 in one place:

- `q1`: IIT Jodhpur corpus collection, preprocessing, Word2Vec training, and semantic analysis
- `q2`: character-level Indian name generation using recurrent neural models

The main submission artifacts, scripts, and generated outputs for both questions are included inside their respective folders.

## Repository Structure

```text
.
├── q1/
│   ├── dataset_generation/   # IITJ web scraping pipeline
│   ├── preprocessing/        # corpus cleaning and normalization
│   ├── model_training/       # CBOW and Skip-gram training
│   ├── inference/            # semantic analysis
│   ├── visualization/        # embedding plots
│   ├── word2vec/             # training utilities
│   ├── output/               # generated corpus, reports, plots, models
│   └── run_pipeline.sh       # end-to-end Q1 pipeline
└── q2/
    ├── dataset/data.txt      # 1000-name training dataset
    ├── run_experiments.py    # trains and compares all Q2 models
    └── results/              # report, metrics, and generated samples
```

## Question 1: Word2Vec on IIT Jodhpur Data

Question 1 builds a domain-specific corpus from IIT Jodhpur web content and trains Word2Vec embeddings on the cleaned text.

### What Q1 Includes

- focused IITJ web scraping and document collection
- preprocessing and normalization of scraped text into a training corpus
- training for both `CBOW` and `Skip-gram` Word2Vec variants
- semantic analysis utilities and embedding visualization
- saved outputs such as processed corpus files, reports, plots, and trained models

### Main Q1 Files

- `q1/run_pipeline.sh`: runs the full Q1 pipeline end to end
- `q1/dataset_generation/run_scraper.py`: collects source documents
- `q1/preprocessing/preprocess.py`: cleans and prepares the corpus
- `q1/model_training/train.py`: trains Word2Vec experiments
- `q1/inference/semantic_analysis.py`: evaluates learned semantics
- `q1/visualization/plot_embeddings.py`: generates plots

### Running Q1

From the repository root:

```bash
bash q1/run_pipeline.sh
```

This pipeline performs:

1. data collection from IITJ sources
2. preprocessing of raw text
3. Word2Vec model training
4. semantic analysis
5. visualization generation

Generated artifacts are written under `q1/output/`.

## Question 2: Character-Level Name Generation

Question 2 trains and compares three character-level sequence models for Indian name generation.

### Models Implemented

- `Vanilla RNN`
- `BLSTM`
- `RNN + Attention`

Each model is trained to predict the next character in a name sequence using `<bos>` and `<eos>` boundary tokens.

### What Q2 Includes

- dataset loading and normalization from `q2/dataset/data.txt`
- shared training and evaluation pipeline for all models
- quantitative comparison using novelty and diversity metrics
- qualitative analysis with representative generated samples
- saved report and sample outputs in `q2/results/`

### Running Q2

From the repository root:

```bash
python q2/run_experiments.py
```

This script trains all three models, generates sample names, computes evaluation metrics, and writes outputs such as:

- `q2/results/report.md`
- `q2/results/results.json`
- model-wise sample files in `q2/results/`

## Environment Notes

- The Q1 pipeline uses the repository-local `.venv` Python interpreter if available; otherwise it falls back to `python3`.
- Q2 requires PyTorch and the standard Python dependencies used in `q2/run_experiments.py`.

## Deliverables Summary

- `q1` contains the complete pipeline and outputs for the IITJ Word2Vec task.
- `q2` contains the full code, evaluation, and generated samples for the character-level name generation task.
- This top-level `README.md` is the single combined README for both questions.
