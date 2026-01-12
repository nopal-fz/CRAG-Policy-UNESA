# CRAG UNESA Academic Policy QA

## Overview

This project implements a **Corrective Retrieval-Augmented Generation (CRAG)** system for question answering over **UNESA Academic Policy (Pedoman Akademik) PDF documents**. The system is designed to provide **evidence-grounded answers**, reduce hallucinations, and explicitly abstain when sufficient evidence is not found.

Unlike simple RAG demos, this project focuses on **engineering reliability** for long, structured, and policy-heavy documents.

---

## Motivation

Initially, this project was planned as a **PEFT fine-tuning** experiment. However, for academic policy documents:

* Creating a high-quality custom QA dataset is costly and time-consuming
* Fine-tuning does not guarantee factual correctness
* Computational cost is relatively high compared to the benefit

CRAG was chosen instead because it:

* Grounds every answer in retrieved evidence
* Explicitly prevents hallucination
* Is more suitable for official policy documents

---

## Data Source

* **Primary data**: Pedoman Administrasi Akademik dan Kelulusan UNESA 2024 (PDF)
* Language: Indonesian
* Characteristics:

  * Long-form document
  * Hierarchical structure (BAB, Section, Subsection)
  * Policy-oriented content requiring strict grounding

---

## System Architecture

**High-level pipeline:**

1. PDF parsing and structure-aware chunking
2. Metadata-enriched chunk indexing (BAB, Section, page range)
3. Hybrid retrieval (Dense Vector + Lexical/BM25)
4. Cross-encoder reranking
5. CRAG gating (rerank + coverage)
6. Evidence-grounded answer generation
7. Abstention when evidence is insufficient

---

## Key Design Decisions

### 1. Structure-Aware Chunking

Chunks are created based on semantic and document structure boundaries rather than fixed token length to avoid context mixing between sections.

### 2. Hybrid Retrieval

Combines:

* Dense vector similarity (semantic match)
* Lexical matching (keyword overlap)

This improves recall for formal policy language.

### 3. Reranking

A cross-encoder reranker is used to ensure the retrieved chunk truly answers the question, not just shares topical similarity.

### 4. CRAG Gate

The system only generates an answer if:

* Rerank score exceeds a minimum threshold
* Evidence coverage is sufficient

Otherwise, the system explicitly responds with *"Tidak ditemukan"* to prevent hallucination.

### 5. Section-Level Context Filtering

Only the **top-ranked section** is passed to the LLM to avoid context contamination across unrelated policy sections.

---

## LLM & Tools

* **LLM**: Ollama (Qwen2.5 7B Instruct)
* **Frameworks**: LangChain, Streamlit
* **Vector Database**: Qdrant
* **Retrieval**: Hybrid (Dense + Lexical)
* **Frontend**: Streamlit

---

## Example Behavior

**Question:**

> Jelaskan mekanisme undur diri di UNESA secara lengkap

**System Behavior:**

* Retrieves section *BAB I – I. Status Undur Diri*
* Filters unrelated sections (e.g., Registrasi, Yudisium)
* Generates a concise, evidence-based answer
* Includes explicit section and page reference

---

## Project Structure

```
crag-unesa-policy-qa/
├── data/
│   ├── policy.pdf
├── scripts/
│   ├── chunk.py
│   ├── ingest_pdf.py
│   ├── inspect_chunk.py
├── src/
│   ├── retrieval/
│   ├── generation/
│   ├── utils/
│   └── prompts/
│   └── indexing/
├── app/
│   └── streamlit_app.py
├── README.md
└── .gitignore
└── docker-compose.yaml
└── requirements.txt
```

---

## Key Takeaways

* Reliable RAG systems are more about **engineering decisions** than model size
* Chunking and metadata design are critical for policy documents
* Preventing hallucination is more important than maximizing answer rate

---

## Disclaimer

This project is built for **educational and portfolio purposes**. The system does not replace official academic information sources.

---

## Author

Naufal Faiz
