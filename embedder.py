import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

def main():
    """
    Loads knowledge_base.csv, encodes texts, builds FAISS index,
    and saves the index and texts.
    """
    csv_path = 'data/knowledge_base.csv'
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    texts = df['formatted_text'].tolist()
    
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding texts... This may take a while.")
    # Encode texts
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    print("Normalizing vectors for IndexFlatIP (Cosine Similarity)...")
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)
    
    dimension = embeddings.shape[1]
    
    print("Building FAISS IndexFlatIP...")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Save FAISS index
    index_path = 'data/mental_health.index'
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")
    
    # Save texts
    texts_path = 'data/texts.pkl'
    with open(texts_path, 'wb') as f:
        pickle.dump(texts, f)
    print(f"Texts saved to {texts_path}")

if __name__ == "__main__":
    main()
