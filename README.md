# Minds on Fire — LLM Mental Health Companion

A production-ready Retrieval-Augmented Generation (RAG) mental health companion app built with Mistral-7B, Streamlit, FAISS, and a custom BERT-based safety classifier.

## Architecture

```text
+----------------+      +------------------+      +-------------------------+
|                |      |                  |      |                         |
|  User Input    +----->+ Safety Classifier+----->+  Emergency Helplines    |
|                |      |  (BERT-based)    |      |  (If crisis detected)   |
+-------+--------+      +---------+--------+      +-------------------------+
        |                         |
        |                         v
        |               +---------+--------+      +-------------------------+
        +-------------->+  RAG Pipeline    +<-----+  FAISS Vector Index     |
                        |  (Mistral-7B)    |      |  (SentenceTransformers) |
                        +---------+--------+      +-----------+-------------+
                                  |                           ^
                                  |                           |
                                  v                           |
                        +---------+--------+      +-----------+-------------+
                        |                  |      |                         |
                        |  Generated Chat  |      |  Counsel-Chat Dataset   |
                        |  Response        |      |  (HuggingFace)          |
                        +------------------+      +-------------------------+
```

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the full setup and launch:
   ```bash
   bash run_all.sh
   ```

## Resume Points

• Fine-tuned a BERT-based crisis detection classifier achieving an F1-score of {F1_SCORE} on test split, enabling real-time routing to emergency helplines.
• Engineered a highly optimized FAISS IndexFlatIP retrieval pipeline with an average latency of {LATENCY} ms per query.
• Maintained high context relevance with an average cosine similarity of {SIMILARITY} across top-k retrieved documents for robust Retrieval-Augmented Generation.
