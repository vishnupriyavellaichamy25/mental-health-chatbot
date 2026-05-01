#!/bin/bash
set -e

echo "Starting setup for Minds on Fire..."

echo "1. Downloading and preparing data..."
python data_loader.py

echo "2. Building FAISS vector index..."
python embedder.py

echo "3. Training Safety Classifier..."
python safety_classifier.py

echo "4. Running Evaluation..."
python evaluate.py

echo "5. Launching Streamlit App..."
streamlit run app.py
