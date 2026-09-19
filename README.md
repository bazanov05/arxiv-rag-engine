# ArXiv RAG Engine: Autonomous Research Assistant

---

## Project Overview

The **ArXiv RAG Engine** is a sophisticated Retrieval-Augmented Generation system designed to provide grounded, citation-backed answers to complex scientific research questions. Unlike traditional keyword-based search engines, this system combines **high-performance C++ text processing**, **fine-tuned neural embeddings**, and **autonomous LLM agents** to navigate the vast landscape of ArXiv scientific papers.

The engine learns a specialized semantic space where papers are clustered by conceptual and citation-level similarity rather than just shared vocabulary. It features a fully autonomous retrieval loop where the AI can iteratively refine its own search queries to find the most relevant evidence before formulating a final response.

---

## Key Features

### 🚀 High-Performance Architecture
*   **C++ Accelerated Preprocessing:** Text normalization and TF-IDF keyword extraction are implemented in **C++** and exposed via **PyBind11**. This eliminates Python-level bottlenecks, enabling high-speed processing of large paper corpuses.
*   **PostgreSQL + pgvector:** Utilizes industry-standard vector storage with **IVFFlat indexing** for sub-millisecond nearest-neighbor retrieval.
*   **Efficient Ingestion:** Leverages PostgreSQL's `COPY` protocol for high-throughput bulk data loading.

### 🧠 Advanced Representation Learning
*   **Base Encoder:** Uses `allenai-specter`, a transformer specifically pretrained on scientific citations and abstracts.
*   **Metric Learning:** Features a custom **PyTorch Projection Head** fine-tuned using **Triplet Loss**.
*   **Deep Semantic Alignment:** The projection head is trained to minimize the distance between papers sharing significant metadata (categories, keywords, citations) while pushing unrelated papers apart.
*   **W&B Integration:** Experiment tracking and hyperparameter optimization (lr, margin, projection_dim) are managed via **Weights & Biases (W&B) Sweeps**.

### 🤖 Autonomous RAG Pipeline
*   **Gemini-Powered Agent:** Integrated with the **Google Gemini API** using an **autonomous tool-calling loop**.
*   **Self-Correcting Retrieval:** The agent can independently decide to "search again" with reformulated queries if the initial results are insufficient or irrelevant.
*   **Grounded Generation:** Enforces strict adherence to retrieved context with mandatory numerical citations (e.g., `[1]`, `[2]`), virtually eliminating hallucinations.
*   **Interactive CLI:** A polished command-line interface with session management and slash-commands (`/k`, `/sources`, `/help`).

---

## Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Language** | Python 3.10+, C++17 |
| **Machine Learning** | PyTorch, Sentence-Transformers, NumPy |
| **LLM / AI** | Google Gemini API (GenAI SDK) |
| **Database** | PostgreSQL, pgvector |
| **Bindings** | PyBind11, CMake |
| **Experimentation** | Weights & Biases (W&B) |
| **Infrastructure** | Docker, Docker Compose |

---

## System Architecture

The pipeline is organized into five distinct phases:

1.  **Ingestion:** Fetching data from the ArXiv public API and performing lexical analysis.
2.  **Normalization:** Cleaning and tokenizing text using the C++ `TextNormalizer`.
3.  **Representation:** Generating 768-dim SPECTER embeddings and projecting them to a learned 256-dim space.
4.  **Indexing:** Building IVFFlat clusters in the vector database.
5.  **Autonomous RAG:** The iterative loop where the agent retrieves context and generates answers.

---

## Deep Dive: Metric Learning

To align the embedding space with scientific relevance, the model is trained using a **Triplet Loss** objective:
$$L = \max(0, d(a, p) - d(a, n) + \text{margin})$$

*   **Anchor (a):** A target research paper.
*   **Positive (p):** A paper sharing a high TF-IDF similarity score and metadata overlap.
*   **Negative (n):** A paper with low semantic similarity.

This forces the projection head to discard surface-level linguistic noise and focus on deep scientific relationships.

---

## Getting Started

### 1. Infrastructure Setup
Spin up the vector database using Docker:
```bash
docker-compose up -d
```

### 2. Build C++ Components
Compile the accelerated preprocessing bindings:
```bash
# Example for normalizer (repeat for tfidf)
cd normalizer
mkdir build && cd build
cmake .. && make
```

### 3. Data Pipeline & Training
Initialize the database and run hyperparameter sweeps:
```bash
# Ingest data, normalize, and load to DB
python -m src.setup

# Train the projection head using W&B Sweeps
python -m src.train

# Project embeddings and build vector indices
python -m src.embed
```

### 4. Launch the Assistant
Run the interactive CLI loop:
```bash
python -m src.main
```

---

## Interactive Commands

While in the CLI session, you can use the following slash-commands:

*   `/k <n>`: Adjust the number of retrieved papers (e.g., `/k 10`).
*   `/sources`: Inspect the full metadata and abstracts of the papers used in the last answer.
*   `/clear`: Clear the terminal screen.
*   `/help`: View the command list.
*   `/exit`: Gracefully shut down the engine and connection pools.
