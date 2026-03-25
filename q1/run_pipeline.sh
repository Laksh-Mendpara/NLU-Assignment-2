#!/bin/bash
set -euo pipefail

Q1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "$Q1_DIR/.." && pwd)"

if [[ -x "$PARENT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$PARENT_DIR/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

cd "$Q1_DIR"

mkdir -p output/data output/pdfs output/models output/plots

echo "=============================================="
echo "Q1 IITJ Word2Vec Pipeline"
echo "=============================================="
echo "Using Python: $PYTHON_BIN"

echo
echo "1. Collecting data"
"$PYTHON_BIN" dataset_generation/run_scraper.py

echo
echo "2. Preprocessing text"
"$PYTHON_BIN" preprocessing/preprocess.py

echo
echo "3. Training models"
"$PYTHON_BIN" model_training/train.py

echo
echo "4. Running semantic analysis"
"$PYTHON_BIN" inference/semantic_analysis.py > output/semantic_analysis_report.txt
cat output/semantic_analysis_report.txt

echo
echo "5. Making plots"
"$PYTHON_BIN" visualization/plot_embeddings.py

echo
echo "Done."
