---
title: Minds on Fire — LLM Mental Health Companion
emoji: 🔥
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: "1.33.0"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
tags:
  - mental-health
  - nlp
  - rag
  - mistral
  - bert
  - faiss
  - streamlit
  - counselling
  - psychoeducation
short_description: >
  A compassionate AI companion for mental health support, powered by
  Mistral-7B RAG and a BERT crisis safety classifier.
---

# 🔥 Minds on Fire — LLM Mental Health Companion

## ⚠️ Important Disclaimer

**This is an AI companion for psychoeducation and peer-style support.
It is NOT a substitute for professional mental health care.**

**Crisis resources (India):**
- **iCall:** 9152987821 *(Mon–Sat, 8 am–10 pm)*
- **Vandrevala Foundation:** 1860-2662-345 *(24/7)*
- **Emergency:** 112

---

## What is this?

Minds on Fire is an open-source AI mental health companion built with:

- **Mistral-7B-Instruct-v0.2** for empathetic, context-aware response generation
- **FAISS + all-MiniLM-L6-v2** for sub-50 ms retrieval from ~2 K real therapy conversations
- **BERT safety classifier** (fine-tuned for crisis detection) with automatic helpline injection
- **Streamlit** chat UI with full multi-turn session memory

## How to use

Simply type how you're feeling in the chat box. The AI will:
1. Understand your concern using semantic search over real counselling transcripts
2. Check your message for signs of distress or crisis
3. Generate a warm, grounded, evidence-based response
4. Automatically provide crisis resources if needed

## Hardware requirements

This Space runs on **GPU** (A10G recommended for Mistral-7B float16).
CPU-only mode is supported but generation will be significantly slower.

## Source code

The full source code, architecture diagram, and documentation are available
in the Files section of this Space.

## Citation / Credits

- Knowledge base: [counsel-chat](https://huggingface.co/datasets/nbertagnolli/counsel-chat) by nbertagnolli
- LLM: [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) by Mistral AI
- Encoder: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
