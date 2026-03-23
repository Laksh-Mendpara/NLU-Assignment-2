#!/bin/bash
set -e

echo "=============================================="
echo " NLU Assignment 2 - Full Pipeline Execution"
echo "=============================================="

# Ensure output directories exist
mkdir -p output/data output/pdfs output/models output/plots

echo "[1/4] Running Dataset Generation (Web Scraper)"
# Assuming the user has already crawled the dataset. If not, uncomment uncomment below:
# python -m dataset_generation.run_scraper -m 50 -c 10 -d 0.5 

echo "\n[2/4] Running Preprocessing & Statistics"
python preprocessing/preprocess.py

echo "\n[3/4] Running Word2Vec Model Training"
python model_training/train.py

echo "\n[4/4] Running Inference & Semantic Analysis"
python inference/semantic_analysis.py > output/semantic_analysis_report.txt
cat output/semantic_analysis_report.txt

echo "\n[5/5] Generating Visualizations"
python visualization/plot_embeddings.py

echo "\n=============================================="
echo " Pipeline Complete! Check the output/ folder."
echo " outputs/wordcloud.png"
echo " outputs/models/..."
echo " outputs/plots/cbow_tsne.png"
echo " outputs/plots/skipgram_tsne.png"
echo "=============================================="
